import random
import argparse
import logging
import numpy as np
import tensorflow as tf
import math
from pathlib import Path

from src.sac import SoftActorCritic
from src.libs.replay_buffer import ReplayBuffer
from src.libs.connector import RealConnector
import pandas as pd


def random_float(low, high):
    return random.random() * (high - low) + low


def save_and_add_history(out_history: Path, row: dict):
    if not out_history.exists():
        row = {key: [item] for key, item in row.items()}
        df = pd.DataFrame(row,
                          columns=['episode', 'metric', 'episode_reward', 'y_target'])
    else:
        df = pd.read_csv(out_history)
        df = df[['episode', 'metric', 'episode_reward', 'y_target']]
        df = df.append(row, ignore_index=True)
    df.to_csv(out_history)


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
parser.add_argument('--batch_size', type=int, default=64,
                    help='minibatch sample size for training')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to run backprop in an episode')
parser.add_argument('--start_steps', type=int, default=0,
                    help='number of global steps before random exploration ends')
parser.add_argument('--model_path', type=str, default='data/models/',
                    help='path to save model')
parser.add_argument('--model_name', type=str,
                    default=f'model_v5',
                    help='name of the saved model')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for future rewards')
parser.add_argument('--polyak', type=float, default=0.005,
                    help='coefficient for polyak averaging of Q network weights')
parser.add_argument('--learning_rate', type=float, default=0.0002,
                    help='learning rate')
parser.add_argument('--model_address', '-md', default=('10.24.1.201', 5000))
parser.add_argument('--model_observation', '-mo', default=4)
parser.add_argument('--model_action_space', '-mas', default=1)
parser.add_argument('--y_target', '-yt', default=1.2)
parser.add_argument('--discretization_step', '-ds', default=0.1)

if __name__ == '__main__':
    args = parser.parse_args()
    # tf.random.set_seed(args.seed)
    # writer = tf.summary.create_file_writer(args.model_path + args.model_name + '2' + '/summary')

    state_space = args.model_observation
    action_space = args.model_action_space

    # Initialize Replay buffer.
    replay = ReplayBuffer(state_space, action_space)

    # Initialize policy and Q-function parameters.
    sac = SoftActorCritic(action_space,
                          learning_rate=args.learning_rate,
                          gamma=args.gamma, polyak=args.polyak)

    sac.load_model(Path(args.model_path), args.model_name)

    print('Weights loaded')

    # Instantiate the environment.
    print('Start connection')
    connector_to_model = RealConnector(args.model_address)
    print('Conn init')

    # Repeat until convergence
    global_step = 1
    episode = 1
    episode_rewards = []
    done = False
    flag = 1
    state_to_output = []
    count_state = 0

    while count_state == 0:

        # Observe state
        current_state = [0.0, 0.0, 0.0, 0.0]

        step = 1
        episode_reward = 0
        y_target = np.random.uniform(0.20, 1.4)
        if global_step != 1:
            connector_to_model.step(0, flag, y_target)
            next_state, metric, done, = connector_to_model.receive(y_target)
            if done == 0:
                flag = 0
        count = 0
        while not done:

            if global_step < args.start_steps:
                if np.random.uniform() > 0.8:
                    action = random_float(0, 10)
                else:
                    action = sac.sample_action(current_state)
            else:
                action = sac.sample_action(current_state)

            # Execute action, observe next state and reward
            connector_to_model.step(action, flag, y_target)
            next_state, metric, done = connector_to_model.receive(y_target)

            y_true = next_state[1]
            E = y_target - y_true
            if E < 0:
                E_norm = E * (1.74 / (1.74 - y_target))
            else:
                E_norm = E * (1.74 / y_target)
            reward = 1.26 * math.exp(-5 * E_norm ** 2) - 0.63

            episode_reward += reward

            # Set end to 0 if the episode ends otherwise make it 1
            # although the meaning is opposite but it is just easier to mutiply
            # with reward for the last step.
            if done:
                end = 0
                flag = 1
                print('count = ', count)
                # count_state = 1
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
            count += 1

        if not end:
            if (step % 1 == 0) and (global_step > args.start_steps):
                for epoch in range(args.epochs):

                    # Randomly sample minibatch of transitions from replay buffer
                    current_states, actions, rewards, next_states, ends = replay.fetch_sample(
                        num_samples=args.batch_size)

                    # Perform single step of gradient descent on Q and policy network
                    critic1_loss, critic2_loss, actor_loss, alpha_loss = sac.train(current_states, actions, rewards,
                                                                                   next_states, ends)
                    # if args.verbose:
                    #     print(episode, global_step, epoch, critic1_loss.numpy(),
                    #           critic2_loss.numpy(), actor_loss.numpy(), episode_reward)

                    sac.epoch_step += 1

                    if sac.epoch_step % 1 == 0:
                        sac.update_weights()

            if episode % 1 == 0:
                sac.save_model(Path(args.model_path), args.model_name + '_dynamic')
            episode_rewards.append(episode_reward)
            row = {'episode': episode, 'metric': current_metric, 'episode_reward': episode_reward, 'y_target': y_target}
            save_and_add_history(Path(args.model_path) / f'{args.model_name}_dynamic_his.csv', row)
            episode += 1
            avg_episode_reward = sum(episode_rewards[-100:]) / len(episode_rewards[-100:])

            print(f"Episode {episode} reward: {episode_reward}")
            print(f"{episode} Average episode reward: {avg_episode_reward}")
            print(f'Final value of metric {current_metric}')
            print(f'Target {y_target}')
            end = 1
