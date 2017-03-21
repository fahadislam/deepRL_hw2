"""Main DQN agent."""
import json 
import os
import numpy as np

from deeprl_hw2.core import Sample

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
    def __init__(self, q_network, preprocessor, memory, policy, gamma,
                 target_update_freq, num_burn_in, train_freq, batch_size,
                 num_actions, log_dir): 

        self.q_network = q_network
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
        self.q_network.compile(loss=loss_func, optimizer=optimizer)

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        # memory.sample(batch_size)
        # Q_sa = np.zeros((len(state), self.num_actions))  
        # for i in range(state.shape(0)):
        # raw_input()
        Q_sa = self.q_network.predict(state)
        return Q_sa

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
            q_values = self.calc_q_values(state)
            action = self.policy.select_action(q_values)
        # else:
        #     self.policy = GreedyEpsilonPolicy(epsilon)
        #     action = policy.select_action(q_values)

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
        minibatch = self.preprocessor.atari.process_batch(minibatch)

        inputs = np.zeros((self.batch_size, minibatch[0].state.shape[1],
                           minibatch[0].state.shape[2],
                           minibatch[0].state.shape[3]))  # 32, 80, 80, 4
        targets = np.zeros((inputs.shape[0], self.num_actions))

        for i in range(0, self.batch_size):
            state_t = minibatch[i].state
            action_t = minibatch[i].action  
            reward_t = minibatch[i].reward
            state_t1 = minibatch[i].next_state
            terminal = minibatch[i].is_terminal

            #inputs[i:i + 1] = state_t
            inputs[i] = state_t 

            targets[i] = self.calc_q_values(state_t)  # Hitting each buttom probability
            Q_sa = self.calc_q_values(state_t1)

            if terminal:
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + self.gamma * np.max(Q_sa)

        # targets2 = normalize(targets)
        loss = self.q_network.train_on_batch(inputs, targets)
        # print("loss", loss)
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
        snapshot_interval = 0 
        while True:
            frame_count = 0 
            acc_reward = 0
            x_t = env.reset()

            s_t = self.preprocessor.process_state_for_network(x_t)
            a_last = -1
            for i in range(max_episode_length):

                # select action
                if a_last >= 0 and frame_count % self.train_freq != 0:
                    a_t = a_last
                else:
                    a_t = self.select_action(s_t)

                # simulate 
                x_t1_colored, r_t, is_terminal, _ = env.step(a_t)  # any action
                acc_reward += r_t
                s_t1 = self.preprocessor.process_state_for_network(x_t1_colored)
                
                frame_count += 1

                # add more into replay memory
                self.memory.append(Sample(s_t, a_t, r_t, s_t1, is_terminal))
                
                s_t = s_t1    # was a bug
                # sample minibatches from replay memory
                if frame_count % self.train_freq != 0 and self.memory.size() >= self.num_burn_in:
                    loss = self.update_policy()

                if self.iterations == num_iterations:
                    print("We've reached the maximum number of iterations... ")
                    return

                if is_terminal:
                    # to be implemented
                    self.memory.end_episode(s_t1, is_terminal)
                    break
                
                self.iterations += 1

            # to be implemented
            self.memory.end_episode(s_t1, is_terminal)  
            print('Iteration', self.iterations, '/ acc reward', acc_reward)

            snapshot_interval += 1
            if snapshot_interval == 5:
                print('Saving model snapshots ... ') 
                model_name = 'model_iter%08d' % self.iterations
                model_path = os.path.join(self.log_dir, model_name)
                self.q_network.save_weights(model_path + '.h5')
                with open(model_path + '.json', "w") as outfile:
                    json.dump(self.q_network.to_json(), outfile)
                snapshot_interval = 0

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
