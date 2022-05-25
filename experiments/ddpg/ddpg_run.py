import logging
import yaml
import tensorflow as tf

from pathlib import Path

from src.libs.dvc_utils import parser_args_for_sac, run_learning, set_connector
from src.ddpg.ddpg import DDPG
from src.libs.replay_buffer import ReplayBuffer


tf.keras.backend.set_floatx('float64')
logging.basicConfig(level='INFO')

if __name__ == '__main__':
    args = parser_args_for_sac()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)

    additional_params = params_all['sac_params'].get('additional_params')
    general_params = params_all['sac_params'].get('general_params')
    neural_network_params = params_all['sac_params'].get('neural_network_params')
    experiment_params = params_all['sac_params'].get('experiment_params')

    tf.random.set_seed(additional_params.get('seed'))

    state_space = general_params['model_observation']
    action_space = general_params['model_action_space']

    # Initialize Replay buffer.
    replay = ReplayBuffer(state_space, action_space)

    # Initialize policy and Q-function parameters.
    ddpg = DDPG(alpha=neural_network_params['alpha'], beta=neural_network_params['beta'], batch_size=general_params['batch_size'], input_dims=general_params['model_observation'])
    # / general_params['model_name']
    full_path = Path(args.model_path)
    history_path = Path(args.output_history_dir)

    if full_path.exists():
        ddpg.load_model()

    full_path.mkdir(exist_ok=True, parents=True)
    history_path.mkdir(exist_ok=True, parents=True)

    connector, do_episode = set_connector(general_params=general_params,
                                          learning_mode=experiment_params.get('learning_mode'))

    run_learning(output_path=full_path, history_path=history_path, rl_model=ddpg, buffer=replay,
                 additional_params=additional_params, general_params=general_params,
                 neural_network_params=neural_network_params, connector=connector,
                 episode_executing_function=do_episode)
