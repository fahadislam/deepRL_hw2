"""Main DQN agent."""
import json 
import os
import copy
import numpy as np
import random
import datetime
import time

from matplotlib import pyplot as plt

from deeprl_hw2.core import Sample

from keras.models import model_from_json

import tensorflow as tf

# from deeprl_hw2.visual import flatten_state

import pdb 


class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """

    # TODO: remove num_actions - instead, derive from q_network
    def __init__(self, sess, env, type, q_source, q_target, preprocessor, memory, policy, policy_eval, 
                 gamma, target_update_freq, num_burn_in, train_freq, batch_size,
                 num_actions, updates_per_epoch, log_dir): 

        self.sess = sess
        self.env = env
        
        self.type = type
        self.q_source = q_source
        self.q_target = q_target
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.policy_eval = policy_eval
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.log_dir = log_dir

        self.num_actions = num_actions
        self.iterations = 0
        self.updates_per_epoch = updates_per_epoch
        self.reset_target_count = 0

        self.summary_writer = tf.summary.FileWriter(os.path.join(log_dir, "tb_log"))
        self.train_summary_placeholders, self.train_assign_ops, self.train_summary_op = self.build_training_summaries()
        self.test_summary_placeholders, self.test_assign_ops, self.test_summary_op = self.build_testing_summaries()
        
    def build_training_summaries(self):
        with tf.name_scope('training'):
            episode_reward = tf.Variable(0.)
            tf.summary.scalar("average reward per life", episode_reward)
            episode_ave_max_q = tf.Variable(0.)
            tf.summary.scalar("average max Q per life", episode_ave_max_q)
            episode_ave_loss = tf.Variable(0.)
            tf.summary.scalar("average loss per life", episode_ave_loss)
            summary_vars = [episode_reward, episode_ave_max_q, episode_ave_loss]
            summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
            assign_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
            summary_op = tf.summary.merge_all()
            self.sess.run(tf.global_variables_initializer())
            return summary_placeholders, assign_ops, summary_op

    def build_testing_summaries(self):
        with tf.name_scope('testing'):
            episode_reward = tf.Variable(0.)
            tf.summary.scalar("average reward per game", episode_reward)
            episode_length = tf.Variable(0.)
            tf.summary.scalar("game length per game", episode_length)
            summary_vars = [episode_reward, episode_length]
            summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
            assign_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
            summary_op = tf.summary.merge_all()

            self.sess.run(tf.global_variables_initializer())
            return summary_placeholders, assign_ops, summary_op

    def sync_networks(self):
        # ops = [self.q_target.weights[i].assign(self.q_source.weights[i]) for i in range(len(self.q_source.weights))]
        # self.sess.run(ops)
        for (layer_source, layer_target) in zip(self.q_source.layers, self.q_target.layers):
            w = layer_source.get_weights()
            layer_target.set_weights(w)
        # pdb.set_trace()

    def compile_networks(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        self.q_source.compile(loss=loss_func, optimizer=optimizer)
        self.q_target.compile(loss=loss_func, optimizer=optimizer)
        self.sync_networks()

    def load_networks(self, model_path):
        # print(model_path + '.json')
        # self.q_source = model_from_json(model_path + '.json')
        self.q_source.load_weights(model_path + '.h5')
        # self.sync_networks()
            
    def calc_q_values(self, state, target):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        assert(state.dtype=='float32')
        assert(np.max(state)>=0 and np.max(state)<=1)        
        if target:
            return self.q_target.predict(state)
        else:
            return self.q_source.predict(state)

    def select_action(self, state, train=True):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """

        if train:
            if self.memory.size() < self.num_burn_in:
                action = np.random.randint(0, self.num_actions)
            else:
                if random.random() <= self.policy.epsilon:
                    action = np.random.randint(0, self.num_actions)
                else:
                    q_values = self.calc_q_values(state, False)
                    action = np.argmax(q_values)
                if self.policy.epsilon > self.policy.end_value:
                    self.policy.epsilon -= self.policy.step
        else:  # for testing
            q_values = self.calc_q_values(state, False)
            action = self.policy_eval.select_action(q_values)
        return action

    def update(self):
        if self.type == 'double-DQN':
            return self.update_policy_double_DQN()
        elif self.type == 'double-Q':
            return self.update_policy_double_Q()
        elif self.type in ['linear', 'DQN', 'duel']:
            return self.update_policy()
        else:
            raise Exception('Type not supported for updating.')
        
    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """

        minibatch = self.memory.sample(self.batch_size)
        minibatch = self.preprocessor.process_batch(minibatch)

        # state_shape:  1 x 4 x 84 x 84
        state_shape = minibatch[0].state.shape
        # batch_shape: 32 x 4 x 84 x 84
        batch_shape = (self.batch_size, state_shape[1], state_shape[2], state_shape[3])
        state_ts = np.zeros(batch_shape, dtype=np.float32)
        state_t1s = np.zeros(batch_shape, dtype=np.float32)

        for i in range(0, self.batch_size):
            state_ts[i] = minibatch[i].state
            state_t1s[i] = minibatch[i].next_state

        targets = self.calc_q_values(state_ts, False)
        Q_hat = self.calc_q_values(state_t1s, True)

        avg_max_q = np.mean(np.max(targets, axis=1))

        for i in range(0, self.batch_size):
            action_t = minibatch[i].action  
            reward_t = minibatch[i].reward
            assert(reward_t in [-1, 0, 1])
            terminal = minibatch[i].is_terminal

            if terminal:
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + self.gamma * np.max(Q_hat[i])

        assert(state_ts.dtype=='float32')
        assert(0 <= np.min(state_ts) and np.max(state_ts) <= 1)
        loss = self.q_source.train_on_batch(state_ts, targets)

        # occasionally update the target network
        self.reset_target_count += 1
        if self.reset_target_count == self.target_update_freq:
            self.reset_target_count = 0
            self.sync_networks()
            self.evaluate()

        return loss, avg_max_q

    # TODO: double-check preprocessing steps
    def update_policy_double_DQN(self):
        # print 'updating policy using Double DQN'
        
        minibatch = self.memory.sample(self.batch_size)
        minibatch = self.preprocessor.process_batch(minibatch)

        # state_shape: 1 x 4 x 84 x 84
        state_shape = minibatch[0].state.shape
        # batch_shape: 32 x 4 x 84 x 84 
        batch_shape = (self.batch_size, state_shape[1], state_shape[2], state_shape[3])
        state_ts = np.zeros(batch_shape, dtype=np.float32)  # 32, 4, 84, 84
        state_t1s = np.zeros(batch_shape, dtype=np.float32)  # 32, 4, 84, 84

        for i in range(0, self.batch_size):
            state_ts[i] = minibatch[i].state
            state_t1s[i] = minibatch[i].next_state

        targets = self.calc_q_values(state_ts, target=False)    # little unsure
        Q_s = self.calc_q_values(state_t1s, target=False)
        Q_s_bar = self.calc_q_values(state_t1s, target=True)
        
        avg_max_q = np.mean(np.max(targets, axis=1))

        for i in range(0, self.batch_size):
            action_t = minibatch[i].action  
            reward_t = minibatch[i].reward
            assert(reward_t in [-1, 0, 1])
            terminal = minibatch[i].is_terminal
            
            if terminal:
                targets[i, action_t] = reward_t
            else:
                a_max = np.argmax(Q_s[i])
                targets[i, action_t] = reward_t + self.gamma * Q_s_bar[i, a_max]

        assert(state_ts.dtype=='float32')
        assert(0 <= np.min(state_ts) and np.max(state_ts) <= 1)        
        loss = self.q_source.train_on_batch(state_ts, targets)

        # occasionally update the target network
        self.reset_target_count += 1
        if self.reset_target_count == self.target_update_freq:
            self.reset_target_count = 0
            self.sync_networks()
            self.evaluate()
        
        return loss, avg_max_q

    # TODO: double-check preprocessing steps
    def update_policy_double_Q(self):
        # print 'updating policy using Double DQN'
        
        minibatch = self.memory.sample(self.batch_size)
        minibatch = self.preprocessor.process_batch(minibatch)

        # state_shape: 1 x 4 x 84 x 84
        state_shape = minibatch[0].state.shape
        # batch_shape: 32 x 4 x 84 x 84 
        batch_shape = (self.batch_size, state_shape[1], state_shape[2], state_shape[3])
        state_ts = np.zeros(batch_shape, dtype=np.float32)  # 32, 4, 84, 84
        state_t1s = np.zeros(batch_shape, dtype=np.float32)  # 32, 4, 84, 84

        for i in range(0, self.batch_size):
            state_ts[i] = minibatch[i].state
            state_t1s[i] = minibatch[i].next_state
        
        coin = random.random() > 0.5

        targets = self.calc_q_values(state_ts, not coin)
        Q_s = self.calc_q_values(state_t1s, not coin)
        Q_s_bar = self.calc_q_values(state_t1s, coin)
        
        avg_max_q = np.mean(np.max(targets, axis=1))

        for i in range(0, self.batch_size):
            action_t = minibatch[i].action  
            reward_t = minibatch[i].reward
            terminal = minibatch[i].is_terminal
            
            if terminal:
                targets[i, action_t] = reward_t
            else:
                a_max = np.argmax(Q_s[i])
                targets[i, action_t] = reward_t + self.gamma * Q_s_bar[i, a_max]
        
        assert(state_ts.dtype=='float32')
        assert(0 <= np.min(state_ts) and np.max(state_ts) <= 1)

        if not coin:
            loss = self.q_source.train_on_batch(state_ts, targets)
        else:
            loss = self.q_target.train_on_batch(state_ts, targets)

        # NOTE: only use this count to trigger model evaluation
        self.reset_target_count += 1
        if self.reset_target_count == self.target_update_freq:
            self.reset_target_count = 0
            # self.sync_networks()
            self.evaluate()

        return loss, avg_max_q

    def update_policy_simple(self, state, action, reward, next_state, is_terminal):
        # minibatch = self.memory.sample(self.batch_size)
        target = self.calc_q_values(state, False)
        Q_hat = self.calc_q_values(next_state, False)
        avg_max_q = np.mean(np.max(target, axis=1))

        if is_terminal:
            target[0,action] = reward
        else: 
            target[0,action] = reward + self.gamma * np.max(Q_hat[0])

        assert(state.dtype=='float32')
        assert(0 <= np.min(state) and np.max(state) <= 1)
        loss = self.q_source.train_on_batch(state, target)

        self.reset_target_count += 1
        if self.reset_target_count == self.target_update_freq:
            self.reset_target_count = 0
            self.evaluate()

        return loss, avg_max_q

        # minibatch = [Sample(state, action, reward, next_state, is_terminal)]
        # minibatch = self.preprocessor.process_batch(minibatch)

        # # state_shape:  1 x 4 x 84 x 84
        # state_shape = minibatch[0].state.shape
        # # batch_shape: 32 x 4 x 84 x 84
        # batch_shape = (self.batch_size, state_shape[1], state_shape[2], state_shape[3])
        # state_ts = np.zeros(batch_shape, dtype=np.float32)
        # state_t1s = np.zeros(batch_shape, dtype=np.float32)

        # for i in range(0, self.batch_size):
        #     state_ts[i] = minibatch[i].state
        #     state_t1s[i] = minibatch[i].next_state

        # targets = self.calc_q_values(state_ts, False)
        # # Q_hat = self.calc_q_values(state_t1s, True)
        # Q_hat = self.calc_q_values(state_t1s, False)

        # avg_max_q = np.mean(np.max(targets, axis=1))

        # for i in range(0, self.batch_size):
        #     action_t = minibatch[i].action  
        #     reward_t = minibatch[i].reward
        #     assert(reward_t in [-1, 0, 1])
        #     terminal = minibatch[i].is_terminal

        #     if terminal:
        #         targets[i, action_t] = reward_t
        #     else:
        #         targets[i, action_t] = reward_t + self.gamma * np.max(Q_hat[i])

        # assert(state_ts.dtype=='float32')
        # assert(0 <= np.min(state_ts) and np.max(state_ts) <= 1)
        # loss = self.q_source.train_on_batch(state_ts, targets)

        # # occasionally update the target network
        # # self.reset_target_count += 1
        # # if self.reset_target_count == self.target_update_freq:
        #     self.reset_target_count = 0
        #     self.sync_networks()
        #     self.evaluate()

        # return loss, avg_max_q

    def fit(self, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """

        # observation = env.render(mode='rgb_array')
        # history_preprocessor = HistoryPreprocessor()
        # sess = tf.InteractiveSession()
        episode_num = 0
        episode_rewards = []
        episode_max_qvalues = []
        episode_loss = []
        episode_length = []
        num_updates = 0
        epoch = 1
        life_num = self.env.unwrapped.ale.lives()
        is_terminal = True

        while True:
            acc_iter = 0
            acc_loss = 0.
            acc_reward = 0.
            acc_qvalue = 0.

            if is_terminal:
                x_t = self.env.reset()
                life_num = self.env.unwrapped.ale.lives()
                
            # s_t = self.preprocessor.process_state_for_network(x_t)
            # a_last = -1
            for i in range(max_episode_length):
                # preprocess the current frame:
                # 1. resize/crop, grayscale, convert to float32,
                # 2. normalize from 255 to 1
                # 3. stack with previous preprocessed frames
                # 4. remove flickering (if needed)
                s_t = self.preprocessor.process_state_for_network(x_t)
                # if i % self.train_freq == 0:
                #     a_t = self.select_action(s_t, train=True)
                # else:
                #     a_t = a_last
                a_t = self.select_action(s_t, train=True)
                # a_last = a_t

                # execute action
                x_t1, r_t, is_terminal, debug_info = self.env.step(a_t)  
                acc_reward += r_t

                # save experience into replay memory
                x_t_to_save = self.preprocessor.process_state_for_memory(x_t)
                self.memory.append(x_t_to_save, a_t, r_t)
                if i == 0:
                    for _ in xrange(self.memory.window_size-1):
                        self.memory.append(x_t_to_save, a_t, r_t)

                # update the current frame
                x_t = x_t1
                
                # NOTE: ignore removing flickering for now
                # if env.env.spec.id.startswith('SpaceInvaders'):
                #     x_t1_fr = self.preprocessor.atari.remove_flickering(x_t, x_t1)
                # else:
                #     x_t1_fr = x_t1
                # # TODO: get rid of preprocessor (handle inside replay mem)
                # s_t1 = self.preprocessor.process_state_for_network(x_t1_colored_fr)
                # add more into replay memory
                # self.memory.append(Sample(s_t, a_t, r_t, s_t1, is_terminal))
                # self.memory.append(s_t, a_t, r_t)
                # if i == 0:
                #     for _ in xrange(self.memory.window_size-1):
                #         self.memory.append(s_t, a_t, r_t)
                        
                # s_t = s_t1    # was a bug
                
                # sample minibatches from replay memory to learn 
                if i % self.train_freq == 0 and self.memory.size() >= self.num_burn_in:
                    loss, avg_max_q = self.update()
                    acc_loss += loss
                    acc_qvalue += avg_max_q
                    num_updates += 1
                    
                if self.iterations == num_iterations:
                    print("We've reached the maximum number of iterations... ")
                    return

                if is_terminal or debug_info['ale.lives'] < life_num:
                    life_num = debug_info['ale.lives']
                    break
                
                self.iterations += 1
                acc_iter += 1
                
                if num_updates > epoch * self.updates_per_epoch:
                    print('Saving model at epoch %d ... ' % epoch)
                    model_path = os.path.join(self.log_dir, 'model_epoch%03d' % epoch)
                    self.q_source.save_weights(model_path + '.h5')
                    with open(model_path + '.json', 'w') as outfile:
                        json.dump(self.q_source.to_json(), outfile)
                    epoch += 1

            # save processed final frame into memory
            # self.memory.end_episode(s_t1, is_terminal)
            x_t1_to_save = self.preprocessor.process_state_for_memory(x_t1)
            self.memory.end_episode(x_t1_to_save, is_terminal)
            self.preprocessor.history.reset()
            episode_num += 1
            
            if self.memory.size() >= self.num_burn_in:
                episode_rewards.append(acc_reward)
                episode_max_qvalues.append(acc_qvalue)
                episode_length.append(acc_iter)
                episode_loss.append(acc_loss)
                ts = time.time()
                st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                print st, ': episode %d, iterations %d, length %d(%d), num_updates %d, max_qvalue %.2f(%.2f), acc_reward %.2f(%.2f) , loss %.3f(%.3f)' % (
                    episode_num, self.iterations, episode_length[-1], np.mean(episode_length), num_updates, acc_qvalue, np.mean(episode_max_qvalues), acc_reward, np.mean(episode_rewards), episode_loss[-1], np.mean(episode_loss))

                stats = [acc_reward, np.mean(episode_max_qvalues), acc_loss]

                feed_dict = {self.train_summary_placeholders[si]:float(stats[si]) for si in range(len(stats))}
                self.sess.run(self.train_assign_ops, feed_dict)
                summary_str = self.sess.run(self.train_summary_op)    
                self.summary_writer.add_summary(summary_str, self.iterations)

    def fit_simple(self, num_iterations, max_episode_length=None):
        episode_num = 0
        episode_rewards = []
        episode_max_qvalues = []
        episode_loss = []
        episode_length = []
        num_updates = 0
        epoch = 1
        life_num = self.env.unwrapped.ale.lives()
        is_terminal = True

        while True:
            acc_iter = 0
            acc_loss = 0.
            acc_reward = 0.
            acc_qvalue = 0.

            if is_terminal:
                x_t = self.env.reset()
                life_num = self.env.unwrapped.ale.lives()
                
            for i in range(max_episode_length):
                # preprocess the current frame:
                # 1. resize/crop, grayscale, convert to float32,
                # 2. normalize from 255 to 1
                # 3. stack with previous preprocessed frames
                # 4. remove flickering (if needed)
                s_t = self.preprocessor.process_state_for_network(x_t)

                # select action
                a_t = self.select_action(s_t, train=True)

                # execute action
                x_t1, r_t, is_terminal, debug_info = self.env.step(a_t)  
                acc_reward += r_t

                # sample minibatches from replay memory to learn 
                if i % self.train_freq == 0 and self.memory.size() >= self.num_burn_in:
                    # loss, max_avg_q = self.update()
                    s_t1 = self.preprocessor.process_state_for_network(x_t1)
                    loss, avg_max_q = self.update_policy_simple(s_t, a_t, r_t, s_t1, is_terminal)
                    acc_loss += loss
                    acc_qvalue += avg_max_q
                    num_updates += 1

                # update the current frame
                x_t = x_t1
                    
                if self.iterations == num_iterations:
                    print("We've reached the maximum number of iterations... ")
                    return

                if is_terminal or debug_info['ale.lives'] < life_num:
                    life_num = debug_info['ale.lives']
                    break
                
                self.iterations += 1
                acc_iter += 1
                
                if num_updates > epoch * self.updates_per_epoch:
                    print('Saving model at epoch %d ... ' % epoch)
                    model_path = os.path.join(self.log_dir, 'model_epoch%03d' % epoch)
                    self.q_source.save_weights(model_path + '.h5')
                    with open(model_path + '.json', 'w') as outfile:
                        json.dump(self.q_source.to_json(), outfile)
                    epoch += 1

            # save processed final frame into memory
            # self.memory.end_episode(s_t1, is_terminal)
            # x_t1_to_save = self.preprocessor.process_state_for_memory(x_t1)
            # self.memory.end_episode(x_t1_to_save, is_terminal)
            self.preprocessor.history.reset()
            episode_num += 1
            
            if self.memory.size() >= self.num_burn_in:
                episode_rewards.append(acc_reward)
                episode_max_qvalues.append(acc_qvalue)
                episode_length.append(acc_iter)
                episode_loss.append(acc_loss)
                ts = time.time()
                st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                print st, ': episode %d, iterations %d, length %d(%d), num_updates %d, max_qvalue %.2f(%.2f), acc_reward %.2f(%.2f) , loss %.3f(%.3f)' % (
                    episode_num, self.iterations, episode_length[-1], np.mean(episode_length), num_updates, acc_qvalue, np.mean(episode_max_qvalues), acc_reward, np.mean(episode_rewards), episode_loss[-1], np.mean(episode_loss))

                stats = [acc_reward, np.mean(episode_max_qvalues), acc_loss]

                feed_dict = {self.train_summary_placeholders[si]:float(stats[si]) for si in range(len(stats))}
                self.sess.run(self.train_assign_ops, feed_dict)
                summary_str = self.sess.run(self.train_summary_op)    
                self.summary_writer.add_summary(summary_str, self.iterations)

    def evaluate(self, episode_num=20, max_episode_length=10000):
        # episode_num = 20
        episode_rewards = np.zeros(episode_num)
        episode_length = np.zeros(episode_num)
        # epoch = 1
        # life_num = env.unwrapped.ale.lives()
        is_terminal = True

        print 'Evaluating', 
        for k in range(episode_num):
            print ',',
            acc_iter = 0
            # acc_loss = 0
            acc_reward = 0.
            # acc_qvalue = 0

            if is_terminal:
                x_t = self.env.reset()
                # life_num = env.unwrapped.ale.lives()
                
            # s_t = self.preprocessor.process_state_for_network(x_t)
            # a_last = -1
            for i in range(max_episode_length):
                # preprocess the current frame:
                # 1. resize/crop, grayscale, convert to float32,
                # 2. normalize from 255 to 1
                # 3. stack with previous preprocessed frames
                # 4. remove flickering (if needed)
                s_t = self.preprocessor.process_state_for_network(x_t)
                # if i % self.train_freq == 0:
                #     a_t = self.select_action(s_t, train=True)
                # else:
                #     a_t = a_last
                # a_last = a_t
                a_t = self.select_action(s_t, train=False)

                # execute action
                x_t1, r_t, is_terminal, debug_info = self.env.step(a_t)  
                acc_reward += r_t

                # # save experience into replay memory
                # x_t_to_save = self.preprocessor.process_state_for_memory(x_t)
                # self.memory.append(x_t_to_save, a_t, r_t)
                # if i == 0:
                #     for _ in xrange(self.memory.window_size-1):
                #         self.memory.append(x_t_to_save, a_t, r_t)

                # update the current frame
                x_t = x_t1
                
                # # sample minibatches from replay memory to learn 
                # if i % self.train_freq == 0 and self.memory.size() >= self.num_burn_in:
                #     loss, max_avg_q = self.update()
                #     acc_loss += loss
                #     acc_qvalue += max_avg_q
                #     num_updates += 1
                    
                # if self.iterations == num_iterations:
                #     print("We've reached the maximum number of iterations... ")
                #     return
                # if is_terminal or debug_info['ale.lives'] < life_num:
                #     life_num = debug_info['ale.lives']
                #     break
                if is_terminal:
                    break
                
                # self.iterations += 1
                acc_iter += 1
                
                # if num_updates > epoch * self.updates_per_epoch:
                #     print('Saving model at epoch %d ... ' % epoch)
                #     model_path = os.path.join(self.log_dir, 'model_epoch%03d' % epoch)
                #     self.q_source.save_weights(model_path + '.h5')
                #     with open(model_path + '.json', 'w') as outfile:
                #         json.dump(self.q_source.to_json(), outfile)
                #     epoch += 1

            # save processed final frame into memory
            # self.memory.end_episode(s_t1, is_terminal)
            # x_t1_to_save = self.preprocessor.process_state_for_memory(x_t1)
            # self.memory.end_episode(x_t1_to_save, is_terminal)
            
            self.preprocessor.history.reset()
            episode_num += 1

            episode_rewards[k] = acc_reward
            episode_length[k] = acc_iter

        stats = [np.mean(episode_rewards), np.mean(episode_length)]
        # for i in range(len(stats)):
        #     self.sess.run(assign_ops[i], {summary_placeholders[i]: float(stats[i])})
        feed_dict = {self.test_summary_placeholders[si]:float(stats[si]) for si in range(len(stats))}
        self.sess.run(self.test_assign_ops, feed_dict)
        summary_str = self.sess.run(self.test_summary_op)    
        self.summary_writer.add_summary(summary_str, self.iterations)

        print ''

    def play(self, episode_num=20, max_episode_length=10000):
        # episode_num = 20
        episode_rewards = np.zeros(episode_num)
        episode_length = np.zeros(episode_num)
        # epoch = 1
        # life_num = env.unwrapped.ale.lives()
        is_terminal = True

        print 'Evaluating', 
        for k in range(episode_num):
            print ',',
            acc_iter = 0
            # acc_loss = 0
            acc_reward = 0.
            # acc_qvalue = 0

            if is_terminal:
                x_t = self.env.reset()
                # life_num = env.unwrapped.ale.lives()
                
            # s_t = self.preprocessor.process_state_for_network(x_t)
            # a_last = -1
            for i in range(max_episode_length):
                # preprocess the current frame:
                # 1. resize/crop, grayscale, convert to float32,
                # 2. normalize from 255 to 1
                # 3. stack with previous preprocessed frames
                # 4. remove flickering (if needed)
                s_t = self.preprocessor.process_state_for_network(x_t)
                # if i % self.train_freq == 0:
                #     a_t = self.select_action(s_t, train=True)
                # else:
                #     a_t = a_last
                # a_last = a_t
                a_t = self.select_action(s_t, train=False)

                # execute action
                x_t1, r_t, is_terminal, debug_info = self.env.step(a_t)
                self.env.render()
                acc_reward += r_t

                # # save experience into replay memory
                # x_t_to_save = self.preprocessor.process_state_for_memory(x_t)
                # self.memory.append(x_t_to_save, a_t, r_t)
                # if i == 0:
                #     for _ in xrange(self.memory.window_size-1):
                #         self.memory.append(x_t_to_save, a_t, r_t)

                # update the current frame
                x_t = x_t1
                
                # # sample minibatches from replay memory to learn 
                # if i % self.train_freq == 0 and self.memory.size() >= self.num_burn_in:
                #     loss, max_avg_q = self.update()
                #     acc_loss += loss
                #     acc_qvalue += max_avg_q
                #     num_updates += 1
                    
                # if self.iterations == num_iterations:
                #     print("We've reached the maximum number of iterations... ")
                #     return
                # if is_terminal or debug_info['ale.lives'] < life_num:
                #     life_num = debug_info['ale.lives']
                #     break
                if is_terminal:
                    break
                
                # self.iterations += 1
                acc_iter += 1
                
                # if num_updates > epoch * self.updates_per_epoch:
                #     print('Saving model at epoch %d ... ' % epoch)
                #     model_path = os.path.join(self.log_dir, 'model_epoch%03d' % epoch)
                #     self.q_source.save_weights(model_path + '.h5')
                #     with open(model_path + '.json', 'w') as outfile:
                #         json.dump(self.q_source.to_json(), outfile)
                #     epoch += 1

            # save processed final frame into memory
            # self.memory.end_episode(s_t1, is_terminal)
            # x_t1_to_save = self.preprocessor.process_state_for_memory(x_t1)
            # self.memory.end_episode(x_t1_to_save, is_terminal)
            
            self.preprocessor.history.reset()
            episode_num += 1

            episode_rewards[k] = acc_reward
            episode_length[k] = acc_iter

        print np.mean(episode_rewards)
        print np.std(episode_rewards)
        # stats = [np.mean(episode_rewards), np.mean(episode_length)]
        # for i in range(len(stats)):
        #     self.sess.run(assign_ops[i], {summary_placeholders[i]: float(stats[i])})
        # feed_dict = {self.test_summary_placeholders[si]:float(stats[si]) for si in range(len(stats))}
        # self.sess.run(self.test_assign_ops, feed_dict)
        # summary_str = self.sess.run(self.test_summary_op)    
        # self.summary_writer.add_summary(summary_str, self.iterations)

        # print ''
        
        # if self.memory.size() >= self.num_burn_in:
        #     episode_rewards.append(acc_reward)
        #     episode_max_qvalues.append(acc_qvalue)
        #     episode_length.append(acc_iter)
        #     episode_loss.append(acc_loss)
        #     ts = time.time()
        #     st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        #     print st, ': episode %d, iterations %d, length %d(%d), num_updates %d, max_qvalue %.2f(%.2f), acc_reward %.2f(%.2f) , loss %.3f(%.3f)' % (
        #         episode_num, self.iterations, episode_length[-1], np.mean(episode_length), num_updates, acc_qvalue, np.mean(episode_max_qvalues), acc_reward, np.mean(episode_rewards), episode_loss[-1], np.mean(episode_loss))

        #     stats = [acc_reward, np.mean(episode_max_qvalues), acc_loss]

        #     for i in range(len(stats)):0
        #         self.sess.run(assign_ops[i],{summary_placeholders[i]: float(stats[i])})
        #         summary_str = self.sess.run(summary_op)    
        #         summary_writer.add_summary(summary_str, self.iterations)                    
