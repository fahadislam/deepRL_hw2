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
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten
from keras.layers import Lambda, Input
from keras.layers.merge import Add
from keras.layers import Conv2D
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras import backend as K

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.core import ReplayMemory, ReplayMemoryEfficient
from deeprl_hw2.policy import LinearDecayGreedyEpsilonPolicy, GreedyEpsilonPolicy
from deeprl_hw2.preprocessors import PreprocessorSequence

import pdb


def create_model_linear(window, input_shape, num_actions, init_method, model_name='q_network'):

    model = Sequential() 

    input_rows, input_cols = input_shape[0], input_shape[1]
    model.add(Flatten(input_shape=(window,input_rows,input_cols)))
    if init_method=='special':
        model.add(Dense(num_actions, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None)))
    elif init_method=='basic':
        model.add(Dense(num_actions))

    return model


def create_model_duel(window, input_shape, num_actions, init_method, model_name='q_network'):  # noqa: D103
    print("Built a dueling sequential DQN")

    input_rows, input_cols = input_shape[0], input_shape[1]

    input = Input(shape=(window, input_rows, input_cols))

    if init_method=='special':
        l_1 = Conv2D(16, kernel_size=(8,8), strides=(4,4), padding='same',
                     kernel_initializer=initializers.he_normal(),
                     activation='relu', input_shape=(window,input_rows,input_cols))(input)
        l_2 = Conv2D(32, kernel_size=(4,4), strides=(2,2), padding='same',
                     kernel_initializer=initializers.he_normal(),
                     activation='relu')(l_1)
        l_3 = Flatten()(l_2)
        v_1 = Dense(128, activation='relu', kernel_initializer=initializers.he_normal())(l_3)
    elif init_method=='basic':
        l_1 = Conv2D(16, kernel_size=(8,8), strides=(4,4), padding='same',
                     activation='relu', input_shape=(window,input_rows,input_cols))(input)
        l_2 = Conv2D(32, kernel_size=(4,4), strides=(2,2), padding='same',
                     activation='relu')(l_1)
        l_3 = Flatten()(l_2)
        v_1 = Dense(128, activation='relu')(l_3)
            
    v_2 = Dense(1)(v_1)
    v_out = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(num_actions,))(v_2)

    if init_method=='special':
        a_1 = Dense(128, activation='relu', kernel_initializer=initializers.he_normal())(l_3)
    elif init_method=='basic':
        a_1 = Dense(128, activation='relu')(l_3)
        
    a_2 = Dense(num_actions)(a_1)
    a_out = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(num_actions,))(a_2)

    # q_out = keras.layers.merge([v_out, a_out], mode='add')
    q_out = Add()([v_out, a_out])
    model = Model(inputs=input, outputs=q_out)

    return model


def create_model(window, input_shape, num_actions, init_method, model_name='q_network'):  # noqa: D103

    input_rows, input_cols = input_shape[0], input_shape[1]

    print 'Now we start building the model ... '
    model = Sequential()
    if init_method=='special':
        model.add(Conv2D(16, kernel_size=(8,8), strides=(4,4), padding='same',
                         kernel_initializer=initializers.he_normal(),
                         activation='relu', input_shape=(window,input_rows,input_cols)))
        model.add(Conv2D(32, kernel_size=(4,4), strides=(2,2), padding='same',
                         kernel_initializer=initializers.he_normal(),
                         activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_initializer=initializers.he_normal()))
        model.add(Dense(num_actions, activation='linear')) 
    elif init_method=='basic':
        model.add(Conv2D(16, kernel_size=(8,8), strides=(4,4), padding='same',
                         activation='relu', input_shape=(window,input_rows,input_cols)))
        model.add(Conv2D(32, kernel_size=(4,4), strides=(2,2), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(num_actions, activation='linear')) 
    elif init_method=='normal':
        model.add(Conv2D(16, kernel_size=(8,8), strides=(4,4), padding='same',
                         kernel_initializer=initializers.truncated_normal(stddev=0.01),
                         activation='relu', input_shape=(window,input_rows,input_cols)))
        model.add(Conv2D(32, kernel_size=(4,4), strides=(2,2), padding='same', activation='relu',
                         kernel_initializer=initializers.truncated_normal(stddev=0.01)))
        model.add(Flatten())
        model.add(Dense(256, kernel_initializer=initializers.random_normal(stddev=0.01), activation='relu'))
        model.add(Dense(num_actions, kernel_initializer=initializers.random_normal(stddev=0.01), activation='linear'))
    return model


def main(args):
    # gpu id
    # gpu_id = args.gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%gpu_id
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
    if args.type in ['DQN', 'double-DQN']:
        model = create_model(window, input_shape, num_actions, args.init)
        target = create_model(window, input_shape, num_actions, args.init)
    elif args.type in ['linear', 'linear-simple', 'double-Q']:
        model = create_model_linear(window, input_shape, num_actions, args.init)
        target = create_model_linear(window, input_shape, num_actions, args.init)
    elif args.type == 'duel':
        model = create_model_duel(window, input_shape, num_actions, args.init)
        target = create_model_duel(window, input_shape, num_actions, args.init)
    # memory = ReplayMemory(1000000, 100)  # window length is arbitrary
    target_update_freq = 10000
    num_burn_in = 500
    train_freq = 4
    batch_size = 32
    gamma = 0.99
    epsilon = 0.05
    updates_per_epoch = 50000
    num_iterations = 50000000
    eval_episodes = 200
    max_episode_length = 10000

    # simple: no experience replay and no target fixing 
    if args.type == 'linear-simple':  
        mem_size = 5
        target_update_freq = 1
        num_burn_in = 0
        batch_size = 1
        
    memory = ReplayMemoryEfficient(mem_size, window, input_shape)
    # with tf.device('/gpu:%d'%gpu_id):
    
    config = tf.ConfigProto(intra_op_parallelism_threads=8)
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
    if args.mode == 'train':  # compile net and train with fit
        rmsprop = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        dqn_agent.compile_networks(rmsprop, mean_huber_loss)
        # adam = Adam(lr=0.00025, beta_1=0.95, beta_2=0.95, epsilon=0.1)
        # dqn_agent.compile_networks(adam, mean_huber_loss)
        dqn_agent.fit(env, num_iterations, max_episode_length)
    elif args.mode == 'test':  # load net and evaluate
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
    parser.add_argument('--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument('--type', default='DQN', type=str, help='DQN|double-DQN|double-Q|duel|linear|linear-simple')
    parser.add_argument('--mode', default='train', help='train|test')
    # parser.add_argument('--run', default=0, type=int, help='run index')
    parser.add_argument('--submit', default=False, type=bool, help='epoch (for test)')
    parser.add_argument('--epoch', default=0, type=int, help='epoch (for test)')
    parser.add_argument('-o', '--output', default='cache', help='Directory to save data to')
    parser.add_argument('--tag', default='', type=str, help='extra tag')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--init', default='normal', type=str, help='normal|special|basic')

    args = parser.parse_args()
    args.output = os.path.join(args.output, '%s-%s'%(args.env, args.type))
    if len(args.tag) > 0:
        args.output = '%s-%s' % (args.output, args.tag)

    # if args.init == 'basic':
    args.output = '%s-%s' % (args.output, args.init)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    return args


if __name__ == '__main__':
    args = parse_input()
    main(args)
