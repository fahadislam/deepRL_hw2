"""Main DQN agent."""
import json 
import os
import copy
import numpy as np
import time
import random

from matplotlib import pyplot as plt

from deeprl_hw2.core import Sample

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
    def __init__(self, q_source, q_target, preprocessor, memory, policy, gamma,
                 target_update_freq, num_burn_in, train_freq, batch_size,
                 num_actions, log_dir): 

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
        self.reset_target_count = 0

        # times
        self.fwd_pass_t = 0;
        self.train_t = 0;
        self.iter_t = 0;

        self.fwd_pass_c = 0;
        self.train_c = 0;
        self.iter_c = 0;

    # NOTE: currently integrated into create_model in dqn_atari.py
    def compile(self, optimizer, loss_func):
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

        for (layer_source, layer_target) in zip(self.q_source.layers, self.q_target.layers):
            w = layer_source.get_weights()
            layer_target.set_weights(w)

    def calc_q_values(self, state, target):
        start_time = time.time()
        self.fwd_pass_c += 1
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        if target:
            Q =  self.q_target.predict(state)
        else:
            Q =  self.q_source.predict(state)
        self.fwd_pass_t += time.time() - start_time
        return Q
        

    def select_action(self, state, **kwargs):
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

        if self.memory.size() < self.num_burn_in:
            action = np.random.randint(0, self.num_actions)
        else:
            if random.random() <= self.policy.epsilon:
                action = np.random.randint(0, self.num_actions)
            else:
                print 'Computing q values'
                q_values = self.calc_q_values(state, False)
                action = np.argmax(q_values)
            if self.policy.epsilon > self.policy.end_value:
                self.policy.epsilon -= self.policy.step
            
        return action

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
        # inputs = np.zeros((self.batch_size, state_shape[1], state_shape[2], state_shape[3]))  # 32, 4, 84, 84
        # targets = np.zeros((self.batch_size, self.num_actions))
        state_ts = np.zeros((self.batch_size, state_shape[1], state_shape[2], state_shape[3]))  # 32, 4, 84, 84
        state_t1s = np.zeros((self.batch_size, state_shape[1], state_shape[2], state_shape[3]))  # 32, 4, 84, 84
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

            # inputs[i] = state_t 
            # targets[i] = self.calc_q_values(state_t, False)  # Hitting each buttom probability
            # Q_hat = self.calc_q_values(state_t1, True)

            if terminal:
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + self.gamma * np.max(Q_hat[i])

        # NOTE: fix training examples and targets to make sure loss is going down

        self.reset_target_count += 1
        if self.reset_target_count == self.target_update_freq:
            self.reset_target_count = 0;
            # weights_target = []
            # for layer in self.q_source.layers:
            for (layer_source, layer_target) in zip(self.q_source.layers, self.q_target.layers):
                # print("LAYER SOURCE")
                # print(layer_source.get_weights())   
                # print("LAYER TARGET")
                # print(layer_target.get_weights())
                w = layer_source.get_weights()
                layer_target.set_weights(w)
        
        # for i in range(10):
        start_time = time.time()
        self.train_c += 1
        # loss = self.q_source.train_on_batch(inputs, targets)
        loss = self.q_source.train_on_batch(state_ts, targets)

        self.train_t += time.time() - start_time
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

        while True:
            
            episode_loss.append(0)
            acc_reward = 0

            x_t = env.reset()

            s_t = self.preprocessor.process_state_for_network(x_t)
            a_last = -1

            self.fwd_pass_t = 0;
            self.train_t = 0;
            self.iter_t = 0;
    
            self.fwd_pass_c = 0;
            self.train_c = 0;
            self.iter_c = 0;
            for i in range(max_episode_length):
                start_time = time.time()
                self.iter_c += 1
                # select action
                if a_last >= 0 and (i+1) % self.train_freq != 0:
                    a_t = a_last
                else:
                    a_t = self.select_action(s_t)

                # simulate 
                x_t1_colored, r_t, is_terminal, _ = env.step(a_t)  # any action
                acc_reward += r_t
                s_t1 = self.preprocessor.process_state_for_network(x_t1_colored)

                # add more into replay memory
                # self.memory.append(Sample(s_t, a_t, r_t, s_t1, is_terminal))
                self.memory.append(s_t, a_t, r_t)
                
                s_t = s_t1    # was a bug
                # sample minibatches from replay memory
                if i % self.train_freq == 0 and self.memory.size() >= self.num_burn_in:
                    loss = self.update_policy()
                    episode_loss[-1] += loss

                if self.iterations == num_iterations:
                    print("We've reached the maximum number of iterations... ")
                    return

                if is_terminal:
                    break
                
                self.iterations += 1
                # print 'iterations: %d, fwd_pass_c: %d' % (self.iterations, self.fwd_pass_c)

                self.iter_t += time.time() - start_time
            print "TIMES"
            print "iter_t", self.iter_t, "train_t", self.train_t, "fwd_pass_t", self.fwd_pass_t
            print "iter_c", self.iter_c, "train_c", self.train_c, "fwd_pass_c", self.fwd_pass_c

            print ('episode', episode_num, 'iterations', self.iterations,
             'acc_reward', acc_reward, 'loss', episode_loss[-1])
            
            # to be implemented
            self.memory.end_episode(s_t1, is_terminal)
            episode_num += 1
            episode_rewards.append(acc_reward)
            
            if episode_num % 20 == 0:
                print('Saving model snapshots at episode %d ... ' % episode_num) 

                model_name = 'model_episode%04d_iter%08d' % (episode_num, self.iterations)
                model_path = os.path.join(self.log_dir, model_name)
                self.q_source.save_weights(model_path + '.h5')
                with open(model_path + '.json', "w") as outfile:
                    json.dump(self.q_source.to_json(), outfile)

                plt.plot(episode_rewards)
                plt.savefig('episode_rewards.png')

            

            

    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        pass
