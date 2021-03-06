"""Suggested Preprocessors."""

import numpy as np
from PIL import Image

from matplotlib import pyplot as plt

from deeprl_hw2 import utils
from deeprl_hw2.core import Sample, Preprocessor

import skimage as skimage
from skimage import transform, color, exposure
import copy

import pdb


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=1):
        self.history_length = history_length
        self.count = 0

    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take."""
        if self.count == 0:
            self.s_t = np.stack((state, state, state, state), axis=0)
            self.s_t = self.s_t.reshape(1, self.s_t.shape[0], self.s_t.shape[1], self.s_t.shape[2])
        else:
            state = state.reshape(1, 1, state.shape[0], state.shape[1])
            self.s_t = np.append(state, self.s_t[:, :self.history_length, :, :], axis=1)
        self.count += 1
        
        return self.s_t

    def reset(self):
        """Reset the history sequence.
        Useful when you start a new episode.
        """
        self.count = 0

    def get_config(self):
        return {'history_length': self.history_length}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size):
        self.new_size = new_size

    def remove_flickering(self, x_t, x_t1):
        return np.maximum(x_t, x_t1)

    def process_frame(self, state):
        # plt.subplot(121)
        # plt.imshow(state)
        state = Image.fromarray(state, 'RGB').convert('LA')
        # state.save('uncropped.png')
        box = (0, 30, 160, 195)
        state = state.crop(box)
        # state.save('cropped.png')
        state = state.resize(self.new_size, Image.BILINEAR)
        # state.save('resized.png')
        state = np.array(state.convert("L"))
        # plt.subplot(122)
        # plt.imshow(state, cmap='gray')
        # plt.show()
        return state
    
    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        state = self.process_frame(state)

        assert(state.dtype=='uint8')
        
        return state

    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """        
        state = self.process_frame(state)
        
        state = state.astype(np.float32)/255.
        assert(state.dtype=='float32')
        
        return state

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        processed_samples = []
        for i in range(len(samples)):
            s_ = samples[i].state.astype(np.float32)/255.
            a = samples[i].action
            r = self.process_reward(samples[i].reward)
            ns_ = samples[i].next_state.astype(np.float32)/255.
            is_t = samples[i].is_terminal
            processed_samples.append(Sample(s_, a, r, ns_, is_t))
            assert(processed_samples[i].state.dtype=='float32')
            assert(processed_samples[i].next_state.dtype=='float32')

        return processed_samples

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        if reward > 0:
            return 1
        elif reward < 0:
            return -1
        else:
            return 0

        
class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreproccessor followed by
    HistoryPreprocessor. This this class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)
    """
    def __init__(self):
        self.atari = AtariPreprocessor((84, 84))
        self.history = HistoryPreprocessor(3)

    def process_state_for_network(self, state):
        # state = self.atari.process_state_for_memory(state)
        state = self.atari.process_state_for_network(state)
        return self.history.process_state_for_network(state)

    def process_state_for_memory(self, state):
        return self.atari.process_state_for_memory(state)

    def process_batch(self, state):
        return self.atari.process_batch(state)
