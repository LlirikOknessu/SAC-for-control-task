import argparse
import logging
from datetime import datetime
import numpy as np
from pathlib import Path
import math

from src.sac import Actor, SoftActorCritic
from src.connector import Connector

if __name__ == '__main__':
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
                        default=f'test',
                        help='name of the saved model')
    parser.add_argument('--model_address', '-md', default=('10.24.1.201', 5000))
    parser.add_argument('--model_observation', '-mo', default=4)
    parser.add_argument('--model_action_space', '-mas', default=1)
    parser.add_argument('--y_target', '-yt', default=1.2)
    parser.add_argument('--discretization_step', '-ds', default=0.1)

    args = parser.parse_args()

    action_space = 1
    sac = SoftActorCritic(action_space, writer=None)

    sac.load_model(Path(args.model_path), args.model_name)
    # Instantiate the environment.
    connector_to_model = Connector(args.model_address)

    while True:
        # Observe state
        current_state = [0.0, 0.0, 0.0, 0.0]

        episode_reward = 0
        done = False
        while not done:

            action = sac.sample_action(current_state)
            # Execute action, observe next state and reward
            connector_to_model.step(action)
            next_state, metric, y_target, done = connector_to_model.receive()

            y_true = next_state[1]
            reward = 1.26 * math.exp(-5 * (y_target - y_true) ** 2) - 0.63

            episode_reward += reward

            if done:
                end = 0
            else:
                end = 1
                current_metric = metric

            # Update current state
            current_state = next_state

        print(episode_reward)
        print('Metric = ', current_metric)
        print('y target was = ', y_target)
