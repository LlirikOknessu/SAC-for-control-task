import argparse
import logging
from datetime import datetime
import numpy as np
import math

from src.sac import Actor
from src.connector import Connector

logging.basicConfig(level='INFO')

parser = argparse.ArgumentParser(description='SAC')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--env_name', type=str, default='MountainCarContinuous-v0',
                    help='name of the gym environment with version')
parser.add_argument('--render', type=bool, default=False,
                    help='set gym environment to render display')
parser.add_argument('--verbose', type=bool, default=False,
                    help='log execution details')
parser.add_argument('--model_path', type=str, default='../data/models/',
                    help='path to save model')
parser.add_argument('--model_name', type=str,
                    default=f'{str(datetime.utcnow().date())}-{str(datetime.utcnow().time())}',
                    help='name of the saved model')
parser.add_argument('--model_address', '-md', default=('192.168.0.122', 5000))
parser.add_argument('--model_observation', '-mo', default=4)
parser.add_argument('--model_action_space', '-mas', default=1)
parser.add_argument('--y_target', '-yt', default=1.2)
parser.add_argument('--discretization_step', '-ds', default=0.1)

while True:

    args = parser.parse_args()


    # Instantiate the environment.
    connector_to_model = Connector(args.model_address)

    state_space = args.model_observation
    action_space = args.model_action_space

    actor = Actor(action_space)

    actor.load_weights(args.model_path + args.model_name + '/model')

    # Observe state
    current_state = [0.0, 0.0, 0.0, 0.0]

    episode_reward = 0
    done = False
    while not done:

        current_state_ = np.array(current_state, ndmin=2)
        action, _ = actor(current_state)

        # Execute action, observe next state and reward
        connector_to_model.step(action)
        next_state, metric, y_target, done = connector_to_model.receive()

        y_true = next_state[1]
        # FIXME fix reward for cases, where y_target is small
        # if y_true > args.y_target:
        # reward = - (2.4 * y_true - 2.88) ** 4
        reward = 1.26 * math.exp(-5 * (y_target - y_true) ** 2) - 0.63

        episode_reward += reward

        # Update current state
        current_state = next_state

    print(episode_reward)
    print('Metric = ', metric)
    print('y target was = ', y_target)
