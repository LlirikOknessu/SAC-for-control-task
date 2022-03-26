import random
import argparse
import logging
import struct

import numpy as np
from datetime import datetime
import tensorflow as tf
import math
from pathlib import Path
from sac_run import save_step_response, save_and_add_history, MOVING_AVERAGE_WINDOW

from src.sac import SoftActorCritic
from src.replay_buffer import ReplayBuffer
from src.connector import Connector
import time


def random_float(low, high):
    return random.random()*(high-low) + low

EPSILON = 1e-7
tf.keras.backend.set_floatx('float64')

logging.basicConfig(level='INFO')

parser = argparse.ArgumentParser(description='SAC')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--env_name', type=str, default='BallTube-v1',
                    help='name of the gym environment with version')
parser.add_argument('--render', type=bool, default=False,
                    help='set gym environment to render display')
parser.add_argument('--verbose', type=bool, default=False,
                    help='log execution details')
parser.add_argument('--batch_size', type=int, default=256,
                    help='minibatch sample size for training')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to run backprop in an episode')
parser.add_argument('--start_steps', type=int, default=10,
                    help='number of global steps before random exploration ends')
parser.add_argument('--model_path', type=str, default='data/models/',
                    help='path to save model')
parser.add_argument('--model_name', type=str,
                    default=f'model_v25_(fixed_target)',
                    help='name of the saved model')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for future rewards')
parser.add_argument('--polyak', type=float, default=0.005,
                    help='coefficient for polyak averaging of Q network weights')
parser.add_argument('--learning_rate', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--model_address', '-md', default=('10.24.1.206', 5000))
parser.add_argument('--model_observation', '-mo', default=4)
parser.add_argument('--model_action_space', '-mas', default=1)
parser.add_argument('--y_target', '-yt', default=1.2)
parser.add_argument('--discretization_step', '-ds', default=0.1)


if __name__ == '__main__':
    args = parser.parse_args()

    tf.random.set_seed(args.seed)

    # Instantiate the environment.
    connector_to_model = Connector(args.model_address)

    state_space = args.model_observation
    action_space = args.model_action_space

    # Initialize Replay buffer.
    replay = ReplayBuffer(state_space, action_space)

    # Initialize policy and Q-function parameters.
    sac = SoftActorCritic(action_space,
                          learning_rate=args.learning_rate,
                          gamma=args.gamma, polyak=args.polyak)

    full_path = Path(args.model_path) / args.model_name
    if full_path.exists():
        sac.load_model(model_name=args.model_name, model_folder=Path(args.model_path))

    # Repeat until convergence
    global_step = 1
    episode = 0
    episode_rewards = []
    done = False
    flag = 1
    state_to_output = []
    count_state = 0
    position = []
    action_list = []
    time_c = []
    current = []
    angular_velocity = []
    object_velocity = []
    learning = True
    moving_average = 0
    y_target = args.y_target
    while True:

        # Observe state
        current_state = [0, 0, 0, 0]

        step = 1
        episode_reward = 0
        done = False
        actions = action_list.copy()
        t0 = time.perf_counter()
        while not done:

            if global_step < args.start_steps:
                if np.random.uniform() > 0.8:
                    action = random_float(0, 10)
                else:
                    action = sac.sample_action(current_state)
            else:
                action = sac.sample_action(current_state)

            # Execute action, observe next state and reward
            t = time.perf_counter() - t0
            connector_to_model.step(action)
            next_state, metric, y_target, done = connector_to_model.receive()

            y_true = next_state[1]
            E = y_target - y_true
            if E < 0:
                E_norm = E * (1.74 / (1.74 - y_target))
            else:
                E_norm = E * (1.74 / y_target)
            reward = 1.26 * math.exp(-5 * E_norm ** 2) - 0.63

            reward *= 2
            episode_reward += reward

            current.append(next_state[0])
            position.append(y_true)
            angular_velocity.append(next_state[2])
            object_velocity.append(next_state[3])
            time_c.append(t)
            if isinstance(action, float):
                actions.append(action)
            else:
                _ = action.numpy()
                actions.append(_[0])

            # Set end to 0 if the episode ends otherwise make it 1
            # although the meaning is opposite but it is just easier to mutiply
            # with reward for the last step.
            if done:
                end = 0
            else:
                end = 1
                current_metric = metric

            if args.verbose:
                logging.info(f'Global step: {global_step}')
                logging.info(f'current_state: {current_state}')
                logging.info(f'action: {action}')
                logging.info(f'reward: {reward}')
                logging.info(f'next_state: {next_state}')
                logging.info(f'end: {end}')

            # Store transition in replay buffer
            replay.store(current_state, action, reward, next_state, end)

            # Update current state
            current_state = next_state

            step += 1
            global_step += 1
        time_c.clear()
        current.clear()
        action_list.clear()
        position.clear()
        angular_velocity.clear()
        object_velocity.clear()

        if (step % 1 == 0) and (global_step > args.start_steps):
            print('Start training')
            episode_rewards.append(episode_reward)

            avg_episode_reward = sum(episode_rewards[-100:]) / len(episode_rewards[-100:])

            print(f"Episode {episode} reward: {episode_reward / 1}")
            print(f"{episode} Average episode reward: {avg_episode_reward / 1}")
            if len(episode_rewards) > MOVING_AVERAGE_WINDOW:
                moving_average = sum(episode_rewards[-MOVING_AVERAGE_WINDOW:]) / MOVING_AVERAGE_WINDOW
                print(f"Episode {episode} moving average reward: {moving_average}")
                if moving_average > 60:
                    learning = False
                    sac.save_model(Path(args.model_path), args.model_name)
                    print('Learning is ended. Best model is saved. \n'
                          f'Model name is {args.model_name}')
                elif moving_average > 30:
                    sac.update_learning_rate(0.0002)
                elif moving_average > 0:
                    sac.update_learning_rate(0.0005)
                elif moving_average > -30:
                    sac.update_learning_rate(0.001)
                elif moving_average > -60:
                    sac.update_learning_rate(0.002)
                elif moving_average > -90:
                    sac.update_learning_rate(0.003)
                elif moving_average > -100:
                    sac.update_learning_rate(0.004)
            print(f'Final value of metric {current_metric}')
            print(f'Target {y_target}')


        if episode % 5 == 0:
            sac.save_model(Path(args.model_path), args.model_name)

        row = {'episode': episode, 'metric': current_metric, 'episode_reward': episode_reward,
               'moving_average': moving_average, 'y_target': y_target}
        save_and_add_history(Path(args.model_path) / f'{args.model_name}_dynamic_his.csv', row)
        episode += 1
