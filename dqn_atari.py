#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import tensorflow as tf
import gym
from gym import wrappers

import keras
from keras import initializers 
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.core import ReplayMemory, ReplayMemoryEfficient
from deeprl_hw2.policy import LinearDecayGreedyEpsilonPolicy, GreedyEpsilonPolicy
from deeprl_hw2.preprocessors import PreprocessorSequence

import pdb


def create_model_large(window, input_shape, num_actions,
                       model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    # Remember you messed up with the initializations

    input_rows, input_cols = input_shape[0], input_shape[1]

    print 'Now we start building the model ... '
    model = Sequential() 
    # model.add(Conv2D(16, kernel_size=(8,8), strides=(4,4), padding='same',
    model.add(Conv2D(32, kernel_size=(8,8), strides=(4,4), padding='same',
                     kernel_initializer=initializers.RandomNormal(stddev=0.01),
                     activation='relu', input_shape=(window,input_rows,input_cols)))
    # model.add(Conv2D(32, kernel_size=(4,4), strides=(2,2), padding='same',
    model.add(Conv2D(64, kernel_size=(4,4), strides=(2,2), padding='same',
                     kernel_initializer=initializers.RandomNormal(stddev=0.01),
                     activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same',
                     kernel_initializer=initializers.RandomNormal(stddev=0.01),
                     activation='relu'))
    model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions, activation='linear')) 

    # plot the architecture of convnet 
    # plot_model(model, to_file='convnet.png')

    return model


def create_model_small(window, input_shape, num_actions,
                       model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    # Remember you messed up with the initializations

    input_rows, input_cols = input_shape[0], input_shape[1]

    print 'Now we start building the model ... '
    model = Sequential() 
    model.add(Conv2D(16, kernel_size=(8,8), strides=(4,4), padding='same',
                     kernel_initializer=initializers.RandomNormal(stddev=0.01),
                     activation='relu', input_shape=(window,input_rows,input_cols)))
    model.add(Conv2D(32, kernel_size=(4,4), strides=(2,2), padding='same',
                     kernel_initializer=initializers.RandomNormal(stddev=0.01),
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_actions, activation='linear')) 

    # plot the architecture of convnet 
    # plot_model(model, to_file='convnet.png')

    return model


def get_output_folder(parent_dir, env_name, mode='train', experiment_id=0):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # find the latest experiment if not assigned
    if experiment_id == 0:
        for folder_name in os.listdir(parent_dir):
            if not os.path.isdir(os.path.join(parent_dir, folder_name)):
                continue
            try:
                folder_name = int(folder_name.split('-run')[-1])
                if folder_name > experiment_id:
                    experiment_id = folder_name
            except:
                pass

    if mode == 'train':
        experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    return parent_dir


def main(args):
    # gpu id
    gpu_id = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%gpu_id
    # make env 
    env = gym.make(args.env)
    if args.mode == 'test' and args.submit:
        monitor_log = os.path.join(args.output, 'monitor.log')
        env = wrappers.Monitor(env, monitor_log)
    # build model
    # actions 0-5: 0 do nothing, 1 fire, 2 right, 3 left, 4 right+fire, 5 left+fire
    num_actions = env.action_space.n
    mem_size = 1000000
    window = 4
    input_shape = (84, 84)
    if args.type == 'normal' or args.type == 'double':
        if args.netsize == 'small':
            model = create_model_small(window, input_shape, num_actions)
            target = create_model_small(window, input_shape, num_actions)
        elif args.netsize == 'large':
            model = create_model_large(window, input_shape, num_actions)
            target = create_model_large(window, input_shape, num_actions)
    # memory = ReplayMemory(1000000, 100)  # window length is arbitrary
    memory = ReplayMemoryEfficient(mem_size, window, input_shape)
    target_update_freq = 10000
    num_burn_in = 1000
    train_freq = 4
    batch_size = 32
    gamma = 0.99
    epsilon = 0.05
    learning_rate = 1e-4
    updates_per_epoch = 50000
    num_iterations = 50000000
    eval_episodes = 200
    max_episode_length = 10000
    with tf.device('/gpu:%d'%gpu_id): 
        config = tf.ConfigProto(intra_op_parallelism_threads=12)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # preprocessor
        preprocessor = PreprocessorSequence()
        # policy
        if args.mode == 'train':
            policy = LinearDecayGreedyEpsilonPolicy(1, 0.1, 1000000)
        elif args.mode == 'test':
            policy = GreedyEpsilonPolicy(epsilon)
        # build agent
        dqn_agent = DQNAgent(args.type, model, target, preprocessor, memory, policy,
                             gamma, target_update_freq, num_burn_in, train_freq,
                             batch_size, num_actions, updates_per_epoch,
                             args.output)
        if args.mode == 'train': # compile net and train with fit
            rmsprop = RMSprop(lr=learning_rate)
            dqn_agent.compile_networks(rmsprop, mean_huber_loss)
            dqn_agent.fit(env, num_iterations, max_episode_length)
        elif args.mode == 'test': # load net and evaluate
            model_path = os.path.join(args.output, 'model_epoch%03d' % args.epoch)
            dqn_agent.load_networks(model_path)
            lengths, rewards = dqn_agent.evaluate(env, eval_episodes, max_episode_length)
            if args.submit:
                gym.upload(monitor_log, api_key='sk_wa5MgeDTnOQ209qBCP7jQ')
            else:
                log_file = open(os.path.join(args.output, 'evaluation.txt'), 'a+')
                log_file.write('%d %f %f %f %f\n' % (args.epoch,
                                                     np.mean(lengths),
                                                     np.std(lengths),
                                                     np.mean(rewards),
                                                     np.std(rewards)))
                log_file.close()
    env.close()


def parse_input():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Space Invaders')
    parser.add_argument('--gpu', default=0, type=int, help='Atari env name')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('--type', default='normal', type=str, help='normal|double|duel')
    parser.add_argument('--mode', default='train', help='train|test')
    parser.add_argument('--netsize', default='small', type=str, help='small|large')
    # parser.add_argument('--run', default=0, type=int, help='run index')
    parser.add_argument('--submit', default=False, type=bool, help='epoch (for test)')
    parser.add_argument('--epoch', default=0, type=int, help='epoch (for test)')
    parser.add_argument('-o', '--output', default='cache', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    # args.output = get_output_folder(args.output, args.env, args.mode, args.run)
    args.output = os.path.join(args.output, '%s-%s-%s'%(args.env, args.type, args.netsize))
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    return args

if __name__ == '__main__':
    args = parse_input()
    main(args)
