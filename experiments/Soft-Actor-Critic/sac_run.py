import logging
import yaml
import tensorflow as tf
from pathlib import Path
from libs.dvc_utils import parser_args_for_sac, run_learning
from sac.sac import SoftActorCritic
from libs.replay_buffer import ReplayBuffer
from libs.connector import Connector

tf.keras.backend.set_floatx('float64')
logging.basicConfig(level='INFO')

if __name__ == '__main__':
    args = parser_args_for_sac()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)

    additional_params = params_all['additional_params']
    general_params = params_all['general_params']
    neural_network_params = params_all['neural_network_params']

    tf.random.set_seed(additional_params.get('seed'))

    # Instantiate the environment.
    connector_to_model = Connector(general_params.get('math_model_address'))

    state_space = general_params['model_observation']
    action_space = general_params['model_action_space']

    # Initialize Replay buffer.
    replay = ReplayBuffer(state_space, action_space)

    # Initialize policy and Q-function parameters.
    sac = SoftActorCritic(action_space,
                          learning_rate=neural_network_params['learning_rate'],
                          gamma=neural_network_params['gamma'], polyak=neural_network_params['polyak'])

    full_path = Path(args.model_path) / general_params['model_name']
    if full_path.exists():
        sac.load_model(model_name=general_params['model_name'], model_folder=Path(args.model_path))

    run_learning(output_path=full_path, rl_model=sac, buffer=replay, additional_params=additional_params,
                 general_params=general_params, neural_network_params=neural_network_params)
