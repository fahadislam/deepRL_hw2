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
    def __init__(self, type, q_source, q_target, preprocessor, memory, policy,
                 gamma, target_update_freq, num_burn_in, train_freq, batch_size,
                 num_actions, updates_per_epoch, log_dir): 

        self.type = type
        self.q_source = q_source
        self.q_target = q_target
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
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

    def sync_networks(self):
        for (layer_source, layer_target) in zip(self.q_source.layers, self.q_target.layers):
            w = layer_source.get_weights()
            layer_target.set_weights(w)

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
        self.sync_networks()
            
    def calc_q_values(self, state, target):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
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
                    # print 'Computing q values'
                    q_values = self.calc_q_values(state, False)
                    action = np.argmax(q_values)
                if self.policy.epsilon > self.policy.end_value:
                    self.policy.epsilon -= self.policy.step
        else:  # for testing
            q_values = self.calc_q_values(state, False)
            action = self.policy.select_action(q_values)

        return action

    def update(self):
        if args.type == 'normal':
            return self.update_policy()
        elif args.type == 'double':
            return self.update_policy_double()
        elif args.type == 'duel':
            raise Exception('Not implemented')
        
    
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

        #print('Updating policy ... ') 

        minibatch = self.memory.sample(self.batch_size)
        # minibatch = self.preprocessor.atari.process_batch(minibatch)

        state_shape = minibatch[0].state.shape
        state_ts = np.zeros((self.batch_size, state_shape[1], state_shape[2], state_shape[3]))  # 32, 4, 84, 84
        state_t1s = np.zeros((self.batch_size, state_shape[1], state_shape[2], state_shape[3]))  # 32, 4, 84, 84
        # targets = np.zeros((self.batch_size, self.num_actions))

        for i in range(0, self.batch_size):
            state_ts[i] = minibatch[i].state
            state_t1s[i] = minibatch[i].next_state
            
        targets = self.calc_q_values(state_ts, False)
        Q_hat = self.calc_q_values(state_t1s, True)
        
        for i in range(0, self.batch_size):
            # state_t = minibatch[i].state
            action_t = minibatch[i].action  
            reward_t = minibatch[i].reward
            # state_t1 = minibatch[i].next_state
            terminal = minibatch[i].is_terminal

            #inputs[i:i + 1] = state_t
            # inputs[i] = state_t 
            # targets[i] = self.calc_q_values(state_t, False)  # Hitting each buttom probability
            # Q_hat = self.calc_q_values(state_t1, True)

            if terminal:
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + self.gamma * np.max(Q_hat[i])

        # occasionally update the target network
        self.reset_target_count += 1
        if self.reset_target_count == self.target_update_freq:
            self.reset_target_count = 0
            self.sync_networks()
        
        # for i in range(10):
        loss = self.q_source.train_on_batch(state_ts, targets)
        # print 'Iteration: %d, Loss: %f' % (i, loss)

        return loss

    def fit(self, env, num_iterations, max_episode_length=None):
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
        episode_num = 0
        episode_rewards = []
        episode_loss = []
        episode_length = []
        num_updates = 0
        epoch = 1

        while True:
            episode_loss.append(0)
            episode_length.append(0)
            acc_reward = 0

            x_t = env.reset()

            s_t = self.preprocessor.process_state_for_network(x_t)
            a_last = -1
            for i in range(max_episode_length):

                # select action
                # if a_last >= 0 and (i+1) % self.train_freq != 0:
                #     a_t = a_last
                # else:
                #     a_t = self.select_action(s_t)
                #     print  i
                if i % self.train_freq == 0:
                    a_t = self.select_action(s_t, target=False)
                else:
                    a_t = a_last
                a_last = a_t

                # simulate 
                x_t1_colored, r_t, is_terminal, _ = env.step(a_t)  # any action
                x_t1_colored_fr = self.preprocessor.atari.remove_flickering(x_t, x_t1_colored)
                
                acc_reward += r_t

                # TODO: get rid of preprocessor (handle inside replay mem)
                s_t1 = self.preprocessor.process_state_for_network(x_t1_colored_fr)

                # add more into replay memory
                # self.memory.append(Sample(s_t, a_t, r_t, s_t1, is_terminal))
                self.memory.append(s_t, a_t, r_t)
                
                s_t = s_t1    # was a bug
                x_t = x_t1_colored
                
                # sample minibatches from replay memory
                if i % self.train_freq == 0 and self.memory.size() >= self.num_burn_in:
                    # loss = self.update_policy()
                    loss = self.update()
                    episode_loss[-1] += loss
                    num_updates += 1
                    
                if self.iterations == num_iterations:
                    print("We've reached the maximum number of iterations... ")
                    return

                if is_terminal:
                    break
                
                self.iterations += 1
                episode_length[-1] += 1
                
                if num_updates > epoch * self.updates_per_epoch:
                    print('Saving model at epoch %d ... ' % epoch)
                    model_path = os.path.join(self.log_dir, 'model_epoch%03d' % epoch)
                    self.q_source.save_weights(model_path + '.h5')
                    with open(model_path + '.json', 'w') as outfile:
                        json.dump(self.q_source.to_json(), outfile)
                    epoch += 1

            # to be implemented
            self.memory.end_episode(s_t1, is_terminal)
            episode_num += 1
            episode_rewards.append(acc_reward)

            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            print st, ': episode %d, iterations %d, length %d(%d), num_updates %d, acc_reward %.2f(%.2f) , loss %.3f(%.3f)' % (
                episode_num, self.iterations, episode_length[-1], np.mean(episode_length), num_updates, acc_reward, np.mean(episode_rewards), episode_loss[-1], np.mean(episode_loss))
            
    def evaluate(self, env, eval_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """

        episode_num = 0
        episode_rewards = []
        episode_lengths = []

        while True:
            episode_lengths.append(0)
            acc_reward = 0

            x_t = env.reset()

            s_t = self.preprocessor.process_state_for_network(x_t)
            a_last = -1
            for i in range(max_episode_length):

                # select action
                if i % self.train_freq == 0:
                    a_t = self.select_action(s_t, target=False)
                else:
                    a_t = a_last
                a_last = a_t

                # simulate 
                x_t1_colored, r_t, is_terminal, _ = env.step(a_t)  # any action
                x_t1_colored_fr = self.preprocessor.atari.remove_flickering(x_t, x_t1_colored)
                acc_reward += r_t

                # get new state
                s_t1 = self.preprocessor.process_state_for_network(x_t1_colored_fr)
                s_t = s_t1    # was a bug
                x_t = x_t1_colored

                episode_lengths[-1] += 1
                
                if is_terminal:
                    break

                self.iterations += 1

                # env.render()
                
            episode_num += 1
            episode_rewards.append(acc_reward)

            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            print st, ': episode %d, iterations %d, length %d(%d), acc_reward %.2f(%.2f)' % (
                episode_num, self.iterations, episode_lengths[-1], np.mean(episode_lengths),
                acc_reward, np.mean(episode_rewards))

            if episode_num == eval_episodes:
                break

        return episode_lengths, episode_rewards
