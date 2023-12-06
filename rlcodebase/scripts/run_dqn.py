import sys
import os
import time
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from rlcodebase.infrastructure.rl_trainer import RLTrainer
from rlcodebase.configs import get_env_kwargs
from rlcodebase.infrastructure.utils import pytorch_utils as ptu
from rlcodebase.agents.dqn_agent import DQNAgent


def main():
    ##################################
    # Get arguments from input
    ##################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str)

    # Env
    parser.add_argument('--env_name', type=str, default='LunarLander-Customizable')
    parser.add_argument('--env_rew_weights', type=float, nargs='*', default=None)
    parser.add_argument('--env_noise_level', type=float, nargs='*', default=None)

    # Batch size
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--tmle_batch_size', type=int, default=500)

    # Update frequencies
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--target_update_freq', type=int, default=3000)

    # Q-learning params
    parser.add_argument('--double_q', action='store_true')
    parser.add_argument('--arch_dim', type=int, default=64)
    parser.add_argument('--update_target_with_tmle', action='store_true')

    # CQL
    parser.add_argument('--add_cql_loss', action='store_true', help='Adds CQL loss to MDQN and EMDQN.')
    parser.add_argument('--cql_alpha', type=float, default=0.2, help='Higher values indicated stronger OOD penalty.')

    # System
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    # Logging
    parser.add_argument('--scalar_log_freq', type=int, default=int(5e3))
    parser.add_argument('--params_log_freq', type=int, default=int(5e3))  # Saves the trained networks
    parser.add_argument('--save_best', action='store_true')

    # Offline learning params
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--buffer_path', type=str, default=None)

    # Data path formatting
    parser.add_argument('--no_weights_in_path', action='store_true')

    args = parser.parse_args()

    # Convert to dictionary
    params = vars(args)

    params['train_batch_size'] = params['batch_size']  # Ensure compatibility

    # Decision booleans
    customize_rew = False if params['env_rew_weights'] is None else True

    if params['offline'] and params['buffer_path'] is None:
        raise Exception('Please provide a buffer_path to enable offline learning.')

    ##################################
    # Set system variables
    ##################################
    # Set device
    ptu.init_gpu(
        use_gpu=not params['no_gpu'],
        gpu_id=params['which_gpu']
    )

    ##################################
    # Create directory for logging
    ##################################
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    if customize_rew and not params['no_weights_in_path']:
        logdir = args.exp_name + '_' + args.env_name \
                + '-'.join([str(w) for w in params['env_rew_weights']]) \
                + '_' + time.strftime('%d-%m-%Y_%H-%M-%S')
    else:
        logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime('%d-%m-%Y_%H-%M-%S')

    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print('\n\n\nLOGGING TO: ', logdir, '\n\n\n')

    ##################################
    # Get env specific arguments
    ##################################
    env_args = get_env_kwargs(params['env_name'])

    for k, v in env_args.items():
        # Don't overwrite the input arguments
        if k not in params:
            params[k] = v

    ##################################
    # Run Q-learning
    ##################################
    params['agent_class'] = DQNAgent
    params['agent_params'] = params

    rl_trainer = RLTrainer(params)
    rl_trainer.run_training_loop(
        params['num_timesteps'],
        collect_policy=rl_trainer.agent.actor,
        eval_policy=rl_trainer.agent.actor
    )


if __name__ == '__main__':
    main()
