import numpy as np

from rlcodebase.infrastructure.replay_buffer import MemoryOptimizedReplayBuffer
from rlcodebase.agents.base_agent import BaseAgent
from rlcodebase.critics.dqn_critic import DQNCritic


class DQNAgent(object):
    def __init__(self, env, agent_params):

        self.env = env

        self.offline = agent_params['offline']

        self.agent_params = agent_params

        # Learning params
        self.batch_size = agent_params['batch_size']
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']
        self.update_target_with_tmle = agent_params['update_target_with_tmle']
        self.tmle_batch_size = agent_params['tmle_batch_size']
        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        # Env params
        if not self.offline:
            self.last_obs = self.env.reset()
            self.num_actions = agent_params['ac_dim']
        else:
            self.last_obs = None  # TODO check that this does not break anything
            self.num_actions = agent_params['ac_dim']

        # Actor/Critic
        self.critic = DQNCritic(agent_params, self.optimizer_spec)
        self.actor = self.critic.get_actor_class()(self.critic)

        # Replay buffer
        lander = agent_params['env_name'].startswith('LunarLander')

        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander)
        self.replay_buffer_idx = None

        # Counters
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        # step_env() has already added the transition to the replay buffer (in case of online learning)

        if self.offline:
            self.replay_buffer.store_offline_data(paths)
        else:
            pass

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """        

        # Store the latest observation ('frame') into the replay buffer
        # The replay buffer used here is 'MemoryOptimizedReplayBuffer' in dqn_utils.py
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        eps = self.exploration.value(self.t)

        # Use epsilon greedy exploration when selecting action
        perform_random_action = (np.random.random() < eps) or (self.t < self.learning_starts)
        if perform_random_action:
            action = self.env.action_space.sample()
        else:
            # HINT: Your actor will take in multiple previous observations ('frames') in order
            # to deal with the partial observability of the environment. Get the most recent 
            # 'frame_history_len' observations using functionality from the replay buffer,
            # and then use those observations as input to your actor. 
            frames = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(frames)
        
        # Take a step in the environment using the action from the policy
        self.last_obs, reward, done, _info = self.env.step(action)

        # Store the result of taking this action into the replay buffer
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        # If taking this step resulted in done, reset the env (and the latest observation)
        if done:
            self.last_obs = self.env.reset()

            if self.agent_params['env_name'].startswith('CartPole'):
                self.last_obs = self.last_obs[0]

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [], [], [], [], [], []

    def calc_prop(self, indices):
        props = []
        for t in indices:
            if t < self.learning_starts:
                props.append(1 / self.agent_params['ac_dim'])
            else:
                props.append(1 - self.exploration.value(t))
        props = np.array(props)

        return props

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n, indices_n):
        log = {}

        if self.t > self.learning_starts \
           and self.t % self.learning_freq == 0 \
           and self.replay_buffer.can_sample(self.batch_size):

            prop_n = self.calc_prop(indices_n)
            log = self.critic.update(
                ob_no, ac_na, next_ob_no, re_n, terminal_n, prop_n
            )

            # Update the target network periodically
            if self.num_param_updates % self.target_update_freq == 0:
                if not self.update_target_with_tmle:
                    self.critic.update_target_network()
                else:
                    # Sample a large batch for tmle
                    ob_mo, ac_ma, re_m, next_ob_mo, terminal_m, indices_m = \
                        self.sample(np.minimum(self.tmle_batch_size, self.replay_buffer.num_in_buffer - 1))

                    # Calc. prop. score
                    prop_m = self.calc_prop(indices_m)

                    # Update the target network
                    self.critic.update_target_network_with_tmle(ob_mo, ac_ma, prop_m)

            self.num_param_updates += 1

        self.t += 1

        return log


class LoadedDQNAgent(BaseAgent):
    def __init__(self, file_path, **kwargs):
        super().__init__(**kwargs)

        self.critic = DQNCritic.load(file_path)
        self.actor = self.critic.get_actor_class()(self.critic)

    def train(self) -> dict:
        pass

    def add_to_replay_buffer(self, paths):
        pass

    def sample(self, batch_size):
        pass

    def save(self, path):
        pass
