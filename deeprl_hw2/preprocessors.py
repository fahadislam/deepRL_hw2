"""Suggested Preprocessors."""

import numpy as np
from PIL import Image

from deeprl_hw2 import utils
from deeprl_hw2.core import Preprocessor
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

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """

        state = Image.fromarray(state, 'RGB').convert('LA')
        state.save('uncropped.png')

        box = (0, 30, 160, 195)
        state = state.crop(box)
        state.save('cropped.png')

        state = state.resize(self.new_size, Image.BILINEAR)
        state.save('resized.png')
        
        # TODO solve flicker problem
        # crop_type = 'bottom'
        # state = Image.fromarray(state, 'RGB').convert('LA')
        # state.save("uncropped.png")

        # state_ratio = state.size[0] / float(state.size[1])
        # ratio = self.new_size[0] / float(self.new_size[1])
        # #The image is scaled/cropped vertically or horizontally depending on the ratio
        # if ratio > state_ratio:
        #     state = state.resize((self.new_size[0], int(round(self.new_size[0] * state.size[1] / state.size[0]))),
        #         Image.ANTIALIAS)
        #     # Crop in the top, middle or bottom
        #     if crop_type == 'bottom':
        #         box = (0, 0, state.size[0], self.new_size[1])
        #     elif crop_type == 'top':
        #         box = (0, state.size[1] - self.new_size[1], state.size[0], state.size[1])
        #     else :
        #         raise ValueError('ERROR: invalid value for crop_type')
        #     state = state.crop(box)
        # state.save('cropped.png')
        # pdb.set_trace()
        # elif ratio < state_ratio:
        #     state = state.resize((int(round(self.new_size[1] * state.size[0] / state.size[1])), self.new_size[1]),
        #         Image.ANTIALIAS)
        #     # Crop in the top, middle or bottom
        #     if crop_type == 'top':
        #         box = (0, 0, self.new_size[0], state.size[1])
        #     elif crop_type == 'bottom':
        #         box = (state.size[0] - self.new_size[0], 0, state.size[0], state.size[1])
        #     else :
        #         raise ValueError('ERROR: invalid value for crop_type')
        #     state = state.crop(box)
        # else :
        #     state = state.resize((self.new_size[0], self.new_size[1]),
        #         Image.ANTIALIAS)
        # If the scale is the same, we do not need to crop
        # state.save("cropped.png")
        state = np.array(state.convert("L"))
        return state

    def process_state_for_network(self, state):                 # NEVER CALLED
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        # TODO solve flicker problem
        crop_type = 'top'
        state = Image.fromarray(state, 'RGB').convert('LA')     #read image to convert to gray
        # state.save("uncropped.png")

        state_ratio = state.size[0] / float(state.size[1])
        ratio = self.new_size[0] / float(self.new_size[1])
        #The image is scaled/cropped vertically or horizontally depending on the ratio
        if ratio > state_ratio:
            state = state.resize((self.new_size[0], int(round(self.new_size[0] * state.size[1] / state.size[0]))),
                Image.ANTIALIAS)
            # Crop in the top, middle or bottom
            if crop_type == 'bottom':
                box = (0, 0, state.size[0], self.new_size[1])
            elif crop_type == 'top':
                box = (0, state.size[1] - self.new_size[1], state.size[0], state.size[1])
            else :
                raise ValueError('ERROR: invalid value for crop_type')
            state = state.crop(box)
        # elif ratio < state_ratio:
        #     state = state.resize((int(round(self.new_size[1] * state.size[0] / state.size[1])), self.new_size[1]),
        #         Image.ANTIALIAS)
        #     # Crop in the top, middle or bottom
        #     if crop_type == 'top':
        #         box = (0, 0, self.new_size[0], state.size[1])
        #     elif crop_type == 'bottom':
        #         box = (state.size[0] - self.new_size[0], 0, state.size[0], state.size[1])
        #     else :
        #         raise ValueError('ERROR: invalid value for crop_type')
        #     state = state.crop(box)
        # else :
        #     state = state.resize((self.new_size[0], self.new_size[1]),
        #         Image.ANTIALIAS)
        # If the scale is the same, we do not need to crop
        # state.save("cropped.png")
        state = np.array(state.convert("L"),'f')    #convert to array and store as float32
        return state

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        samples_copy = copy.deepcopy(samples)
        for sample in samples_copy:
            # print "before", sample.state.shape, sample.state.dtype
            sample.state = sample.state.astype(np.float32)
            # print "after", sample.state.shape, sample.state.dtype
            sample.next_state = sample.next_state.astype(np.float32)

        return samples_copy

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        # NOTE: updated by peiyun, based on DQN paper 
        if reward > 0:
            reward = 1
            #reward = min(1, reward)
        else:
            reward = -1
            #reward = max(-1, reward)


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
        state = self.atari.process_state_for_memory(state)
        return self.history.process_state_for_network(state)
