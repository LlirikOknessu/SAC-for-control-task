import random
import argparse
import logging
import struct

import numpy as np
from datetime import datetime
import tensorflow as tf
import math
from pathlib import Path

from src.sac import SoftActorCritic
from src.replay_buffer import ReplayBuffer
from src.connector import Connector


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
                    default=f'model_v19',
                    help='name of the saved model')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for future rewards')
parser.add_argument('--polyak', type=float, default=0.005,
                    help='coefficient for polyak averaging of Q network weights')
parser.add_argument('--learning_rate', type=float, default=0.008,
                    help='learning rate')
parser.add_argument('--model_address', '-md', default=('10.24.1.206', 5000))
parser.add_argument('--model_observation', '-mo', default=5)
parser.add_argument('--model_action_space', '-mas', default=1)
parser.add_argument('--y_target', '-yt', default=1.2)
parser.add_argument('--discretization_step', '-ds', default=0.1)


if __name__ == '__main__':
    args = parser.parse_args()

    # tf.random.set_seed(args.seed)
    #writer = tf.summary.create_file_wr b
    # iter(args.model_path + args.model_name + '/summary')

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

    # sac.policy.load_weights(args.model_path + '/2020-05-30-19:03:13.833421/model')
    full_path = Path(args.model_path) / args.model_name
    if full_path.exists():
        sac.load_model(model_name=args.model_name, model_folder=Path(args.model_path))

    # Repeat until convergence
    global_step = 1
    episode = 1
    episode_rewards = []
    done = False
    while True:

        # Observe state
        current_state = [0, 0, 0, 0, 0]

        step = 1
        episode_reward = 0
        done = False
        while not done:

            if global_step < args.start_steps:
                if np.random.uniform() > 0.8:
                    action = random_float(0, 10)
                else:
                    action = sac.sample_action(current_state)
            else:
                action = sac.sample_action(current_state)

            # Execute action, observe next state and reward
            connector_to_model.step(action)
            try:
                next_state, metric, y_target, done = connector_to_model.receive()
            except struct.error as e:
                print(episode, global_step, epoch, critic1_loss.numpy(),
                      critic2_loss.numpy(), actor_loss.numpy(), episode_reward)
                print(current_states, '\n',
                      next_states, '\n',
                      ends, '\n',
                      actions, '\n',
                      rewards)
                print(e)
                import csv

                fieldnames = ['current_states', 'actions', 'rewards', 'next_states', 'ends']
                with open(f'data/actions_and_rewards_of_{args.model_name}.csv', 'w') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    for current_state, action, reward, next_state, end in zip(replay.current_states, replay.actions, replay.rewards, replay.next_states, replay.ends):
                        writer.writerow([action, reward, end])
                with open(f'data/current_states_of_{args.model_name}.csv', 'w') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    writer.writerows(replay.current_states)
                with open(f'data/next_states_of_{args.model_name}.csv', 'w') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    writer.writerows(replay.next_states)


            y_true = next_state[1]
            E = y_target - y_true
            if E < 0:
                E_norm = E * (1.74 / (1.74 - y_target))
            else:
                E_norm = E * (1.74 / y_target)
            reward = 1.26 * math.exp(-5 * E_norm ** 2) - 0.63

            reward *= 1
            episode_reward += reward

            # Set end to 0 if the episode ends otherwise make it 1
            # although the meaning is opposite but it is just easier to mutiply
            # with reward for the last step.
            if done:
                end = 0
            else:
                end = 1
                current_metric = metric

            #if args.verbose:
            #    logging.info(f'Global step: {global_step}')
            #    logging.info(f'current_state: {current_state}')
            #    logging.info(f'action: {action}')
            #    logging.info(f'reward: {reward}')
            #    logging.info(f'next_state: {next_state}')
            #    logging.info(f'end: {end}')

            # Store transition in replay buffer
            replay.store(current_state, action, reward, next_state, end)

            # Update current state
            current_state = next_state

            step += 1
            global_step += 1

        if (step % 1 == 0) and (global_step > args.start_steps):
            print('Start training')
            for epoch in range(args.epochs):

                # Randomly sample minibatch of transitions from replay buffer
                current_states, actions, rewards, next_states, ends = replay.fetch_sample(num_samples=args.batch_size)
                # print(current_states, actions, rewards, next_states, ends)

                # Perform single step of gradient descent on Q and policy
                # network
                critic1_loss, critic2_loss, actor_loss, alpha_loss = sac.train(current_states, actions, rewards,
                                                                               next_states, ends)
                if args.verbose:
                    print(episode, global_step, epoch, critic1_loss.numpy(),
                          critic2_loss.numpy(), actor_loss.numpy(), episode_reward)

                #with writer.as_default():
                    # tf.summary.scalar("actor_loss", actor_loss, sac.epoch_step)
                #    tf.summary.scalar("critic1_loss", critic1_loss, sac.epoch_step)
                #    tf.summary.scalar("critic2_loss", critic2_loss, sac.epoch_step)
                #    tf.summary.scalar("alpha_loss", alpha_loss, sac.epoch_step)

                sac.epoch_step += 1

                # if sac.epoch_step % 1 == 0:
                #     sac.update_weights()

        if episode % 5 == 0:
            sac.save_model(Path(args.model_path), args.model_name)

        episode_rewards.append(episode_reward)
        episode += 1
        avg_episode_reward = sum(episode_rewards[-100:]) / len(episode_rewards[-100:])

        print(f"Episode {episode} reward: {episode_reward / 1}")
        print(f"{episode} Average episode reward: {avg_episode_reward / 1}")
        print(f'Final value of metric {current_metric}')
        print(f'Target {y_target}')
        #with writer.as_default():
        #    tf.summary.scalar("episode_reward", episode_reward, episode)
        #    tf.summary.scalar("avg_episode_reward", avg_episode_reward, episode)