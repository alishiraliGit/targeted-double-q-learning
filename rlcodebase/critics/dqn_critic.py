import numpy as np
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn

from rlcodebase.infrastructure.utils import pytorch_utils as ptu
from rlcodebase.critics.base_critic import BaseCritic
import rlcodebase.policies.argmax_policy as argmax_policy


class DQNCritic(BaseCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)

        self.hparams = hparams

        # Env
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.re_dim = hparams.get('re_dim', 1)
        self.ex_dim = hparams.get('ex_dim', 1)
        self.arch_dim = hparams.get('arch_dim', 64)

        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        # Networks
        network_initializer = hparams['q_func']
        self.q_net = network_initializer(self.ob_dim, self.ac_dim, self.re_dim, self.ex_dim, self.arch_dim)
        self.q_net_target = network_initializer(self.ob_dim, self.ac_dim, self.re_dim, self.ex_dim, self.arch_dim)

        # Optimization
        self.optimizer_spec = optimizer_spec
        self.optimizer = self.optimizer_spec.constructor(
            self.q_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

        self.loss = nn.SmoothL1Loss()  # AKA Huber loss

        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)

        # CQL
        self.add_cql_loss = hparams.get('add_cql_loss', False)
        if self.add_cql_loss:
            self.cql_alpha = hparams['cql_alpha']

        # TMLE
        self.tmle_eps = np.zeros((1,))

    def get_actor_class(self):
        return argmax_policy.ArgMaxPolicy

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n, prop_n):
        """
            Update the parameters of the critic.
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end
            returns:
                nothing
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)
        prop_n = ptu.from_numpy(prop_n)

        qa_t_values = self.q_net(ob_no)
        q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)
        
        # Compute the Q-values from the target network
        qa_tp1_values = self.q_net_target(next_ob_no)

        if self.double_q:
            # In double Q-learning, the best action is selected using the Q-network that
            # is being updated, but the Q-value for this action is obtained from the
            # target Q-network.
            ac_tp1 = self.q_net(next_ob_no).argmax(dim=1)
            q_tp1 = torch.gather(qa_tp1_values, 1, ac_tp1.unsqueeze(1)).squeeze(1)

            # TMLE adjustment
            if self.hparams['update_target_with_tmle']:
                h_n = 1/prop_n
                q_tp1 = self.itransform_q(torch.sigmoid(
                    torch.logit(self.transform_q(q_tp1)) + ptu.from_numpy(self.tmle_eps)*h_n
                ))
        else:
            q_tp1, _ = qa_tp1_values.max(dim=1)

        # Compute targets for minimizing Bellman error
        # currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        target = reward_n + self.gamma*q_tp1*(1 - terminal_n)
        target = target.detach()

        assert q_t_values.shape == target.shape
        loss = self.loss(q_t_values, target)

        # Add CQL loss if requested
        if self.add_cql_loss:
            q_t_logsumexp = torch.logsumexp(qa_t_values, dim=1)
            cql_loss = torch.mean(q_t_logsumexp - q_t_values)
            loss = self.cql_alpha * cql_loss + loss

        # Step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        self.learning_rate_scheduler.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    @staticmethod
    def transform_q(q):
        return torch.clamp(q / 300 + 0.5, 0.01, 0.99)

    @staticmethod
    def itransform_q(q):
        return (q - 0.5)*300

    def update_target_network_with_tmle(self, ob_no, ac_n, prop_n):
        ob_no = ptu.from_numpy(ob_no)
        ac_n = ptu.from_numpy(ac_n).to(torch.long)
        prop_n = ptu.from_numpy(prop_n)

        # Get Q-values
        with torch.no_grad():
            qa_tar_values = self.q_net_target(ob_no)
            q_tar_values = torch.gather(qa_tar_values, 1, ac_n.unsqueeze(1)).squeeze(1)

            qa_values = self.q_net(ob_no)
            q_values = torch.gather(qa_values, 1, ac_n.unsqueeze(1)).squeeze(1)

        # Find optimal actions
        ac_rule_n = qa_values.argmax(dim=1)

        # Find clever covariate
        msk = (ac_rule_n == ac_n)
        if torch.sum(msk) < 10:
            self.update_target_network()
            return

        h_n = 1/prop_n[msk]

        # Solve for eps
        eps = torch.zeros(1, requires_grad=True)

        y = self.transform_q(q_tar_values[msk])

        loss_func = torch.nn.BCELoss()
        optimizer = optim.SGD([eps], lr=0.01)

        n_iter = 1000

        for it in range(n_iter):
            # Forward pass
            y_pred = torch.sigmoid(torch.logit(self.transform_q(q_values[msk])) + eps*h_n)

            # Compute the loss
            loss = loss_func(y_pred, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Updates
        self.tmle_eps = ptu.to_numpy(eps)
        self.update_target_network()

    def qa_values(self, obs:  np.ndarray) -> np.ndarray:
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)

    def save(self, save_path):
        torch.save(
            {
                'hparams': self.hparams,
                'optimizer_spec': self.optimizer_spec,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'q_net_state_dict': self.q_net.state_dict(),
                'q_net_target_state_dict': self.q_net_target.state_dict()
            }, save_path)

    @classmethod
    def load(cls, load_path):
        checkpoint = torch.load(load_path)

        dqn_critic = cls(
            hparams=checkpoint['hparams'],
            optimizer_spec=checkpoint['optimizer_spec']
        )

        dqn_critic.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        dqn_critic.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        dqn_critic.q_net_target.load_state_dict(checkpoint['q_net_target_state_dict'])

        dqn_critic.q_net.to(ptu.device)
        dqn_critic.q_net_target.to(ptu.device)

        return dqn_critic
