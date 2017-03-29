"""Core classes."""
import random
from collections import deque

import numpy as np
import operator

import pdb


class Sample:
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Note: This is not the most efficient way to store things in the
    replay memory, but it is a convenient class to work with when
    sampling batches, or saving and loading samples while debugging.

    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """
    def __init__(self, state, action, reward, next_state, is_terminal):
      self.state = state    # [1,width,height,1]
      self.action = action
      self.reward = reward
      self.next_state = next_state
      self.is_terminal = is_terminal    


class Preprocessor:
    """Preprocessor base class.

    This is a suggested interface for the preprocessing steps. You may
    implement any of these functions. Feel free to add or change the
    interface to suit your needs.

    Preprocessor can be used to perform some fixed operations on the
    raw state from an environment. For example, in ConvNet based
    networks which use image as the raw state, it is often useful to
    convert the image to greyscale or downsample the image.

    Preprocessors are implemented as class so that they can have
    internal state. This can be useful for things like the
    AtariPreproccessor which maxes over k frames.

    If you're using internal states, such as for keeping a sequence of
    inputs like in Atari, you should probably call reset when a new
    episode begins so that state doesn't leak in from episode to
    episode.
    """

    def process_state_for_network(self, state):
        """Preprocess the given state before giving it to the network.

        Should be called just before the action is selected.

        This is a different method from the process_state_for_memory
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory is a lot more efficient thant float32, but the
        networks work better with floating point images.

        Parameters
        ----------
        state: np.ndarray
          Generally a numpy array. A single state from an environment.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in anyway.

        """
        return state

    def process_state_for_memory(self, state):
        """Preprocess the given state before giving it to the replay memory.

        Should be called just before appending this to the replay memory.

        This is a different method from the process_state_for_network
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory and the network expecting images in floating
        point.

        Parameters
        ----------
        state: np.ndarray
          A single state from an environmnet. Generally a numpy array.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in any manner.

        """
        return state

    def process_batch(self, samples):
        """Process batch of samples.

        If your replay memory storage format is different than your
        network input, you may want to apply this function to your
        sampled batch before running it through your update function.

        Parameters
        ----------
        samples: list(tensorflow_rl.core.Sample)
          List of samples to process

        Returns
        -------
        processed_samples: list(tensorflow_rl.core.Sample)
          Samples after processing. Can be modified in anyways, but
          the list length will generally stay the same.
        """
        return samples

    def process_reward(self, reward):
        """Process the reward.

        Useful for things like reward clipping. The Atari environments
        from DQN paper do this. Instead of taking real score, they
        take the sign of the delta of the score.

        Parameters
        ----------
        reward: float
          Reward to process

        Returns
        -------
        processed_reward: float
          The processed reward
        """
        return reward

    def reset(self):
        """Reset any internal state.

        Will be called at the start of every new episode. Makes it
        possible to do history snapshots.
        """
        pass


class ReplayMemoryEfficient:

    def __init__(self, max_size, window_size, frame_size):
        self.max_size = max_size
        self.window_size = window_size
        self.frame_size = frame_size
        self.index = 0
        self.full = False
        
        self.frames = np.zeros((max_size, frame_size[0], frame_size[1]), dtype=np.uint8)
        self.actions = np.zeros(max_size, dtype=np.uint8)
        self.rewards = np.zeros(max_size, dtype=np.int8)
        self.terminals = np.zeros(max_size, dtype=np.bool_)

    def size(self):
        if self.full:
            return self.max_size
        else:
            return self.index
        
    def check(self):
        if self.index == self.max_size:
            self.full = True
            self.index = 0

    # def append(self, state, action, reward):
    #     self.frames[self.index, :, :] = state[0][0]
    def append(self, frame, action, reward):
        self.frames[self.index, :, :] = frame
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.terminals[self.index] = False
        self.index += 1
        self.check()
        
    # def end_episode(self, state, is_terminal):  # is_terminal has no effect
    #     self.frames[self.index, :, :] = state[0][0]
    def end_episode(self, frame, is_terminal):  # is_terminal has no effect
        self.frames[self.index, :, :] = frame
        self.actions[self.index] = 0
        self.rewards[self.index] = 0
        self.terminals[self.index] = True
        self.index += 1
        self.check()

    def fetch_window(self, indices):
        indices = indices[::-1]
        # bug
        # state = self.frames[indices].astype(np.float32)/255.0
        window = self.frames[indices]
        return window.reshape(1, window.shape[0], window.shape[1], window.shape[2])

    def draw(self, batch_size):
        cnt = 0
        I = -np.ones(batch_size,dtype=np.int)
        while cnt < batch_size:
            if self.full:
                i = np.random.randint(self.window_size-1, self.max_size-1)
                # NOTE: don't sample around the current index
                if self.index - 1 <= i and i <= self.index + 2:
                    continue
            else:
                i = np.random.randint(self.window_size-1, self.index)
            if np.any(self.terminals[i-(self.window_size-1):i+1]):
                continue
            I[cnt] = i
            cnt += 1

        return I
    
    def sample(self, batch_size):
        I = self.draw(batch_size)
        return [Sample(self.fetch_window(np.arange(i-(self.window_size-1), i+1)), 
                       self.actions[i], self.rewards[i],
                       self.fetch_window(np.arange(i-(self.window_size-2), i+2)), 
                       self.terminals[i+1]) for i in I]

    def clear(self):
        self.full = False
        self.index = 0


class ReplayMemory:
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just ranomly draw saamples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), or if it
      is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """

    def __init__(self, max_size, window_length):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.max_size = max_size
        self.index = 0
        # self.window_length = window_length
        self._data = []

    def size(self):
        return len(self._data)

    # def append(self, state, action, reward):
    def append(self, sample):   
        # raise NotImplementedError('This method should be overridden')
        if len(self._data) == self.max_size:
            self._data[self.index]= sample
        else:
            self._data.append(sample)
            
        #self.index= (self.index + 1) % self.max_size
        if self.index == self.max_size - 1:
            self.index = 0
                                
    def end_episode(self, final_state, is_terminal):
        # raise NotImplementedError('This method should be overridden')
        pass

    def sample(self, batch_size, indexes=None):
        I = np.random.randint(0, len(self._data), batch_size)
        return [self._data[i] for i in I]

    def clear(self):
        raise NotImplementedError('This method should be overridden')
