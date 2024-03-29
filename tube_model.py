import random
import argparse
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
import tensorflow as tf
import math
import pickle

from src.sac import SoftActorCritic
from src.replay_buffer import ReplayBuffer
from src.connector import Connector


def random_float(low, high):
    return random.random()*(high-low) + low


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
parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch sample size for training')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of epochs to run backprop in an episode')
parser.add_argument('--start_steps', type=int, default=15,
                    help='number of global steps before random exploration ends')
parser.add_argument('--model_path', type=str, default='data\\models\\',
                    help='path to save model')
parser.add_argument('--model_name', type=str,
                    default='first_try',
                    help='name of the saved model')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for future rewards')
parser.add_argument('--polyak', type=float, default=0.005,
                    help='coefficient for polyak averaging of Q network weights')
parser.add_argument('--learning_rate', type=float, default=0.0003,
                    help='learning rate')
parser.add_argument('--model_address', '-md', default=('192.168.0.123', 5000))
parser.add_argument('--model_observation', '-mo', default=3)
parser.add_argument('--model_action_space', '-mas', default=1)
parser.add_argument('--y_target', '-yt', default=1.2)
parser.add_argument('--discretization_step', '-ds', default=0.1)


if __name__ == '__main__':
    args = parser.parse_args()

    # tf.random.set_seed(args.seed)
    writer = tf.summary.create_file_writer(args.model_path + args.model_name + '/summary')

    # Instantiate the environment.
    connector_to_model = Connector(args.model_address)

    state_space = args.model_observation
    action_space = args.model_action_space

    # Initialize Replay buffer.
    replay = ReplayBuffer(state_space, action_space)

    # Initialize policy and Q-function parameters.
    sac = SoftActorCritic(action_space, writer,
                          learning_rate=args.learning_rate,
                          gamma=args.gamma, polyak=args.polyak)

    # sac.policy.load_weights(args.model_path + '/2020-05-30-19:03:13.833421/model')

    # Repeat until convergence
    global_step = 1
    episode = 1
    episode_rewards = []
    final_metrics = []
    while True:

        # Observe state
        current_state = [0.0, 0.0, 0.0]

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
            next_state, metric, done = connector_to_model.receive()

            y_true = next_state[1]
            #FIXME fix reward for cases, where y_target is small
            reward = 1.26 * math.exp(-5 * (args.y_target - y_true) ** 2) - 0.63

            episode_reward += 2 * reward

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

        if (step % 1 == 0) and (global_step > args.start_steps):
            for epoch in range(args.epochs):

                # Randomly sample minibatch of transitions from replay buffer
                current_states, actions, rewards, next_states, ends = replay.fetch_sample(num_samples=args.batch_size)

                # Perform single step of gradient descent on Q and policy
                # network
                critic1_loss, critic2_loss, actor_loss, alpha_loss = sac.train(current_states, actions, rewards,
                                                                               next_states, ends)
                if args.verbose:
                    print(episode, global_step, epoch, critic1_loss.numpy(),
                          critic2_loss.numpy(), actor_loss.numpy(), episode_reward)

                with writer.as_default():
                    tf.summary.scalar("actor_loss", actor_loss, sac.epoch_step)
                    tf.summary.scalar("critic1_loss", critic1_loss, sac.epoch_step)
                    tf.summary.scalar("critic2_loss", critic2_loss, sac.epoch_step)
                    tf.summary.scalar("alpha_loss", alpha_loss, sac.epoch_step)

                sac.epoch_step += 1

                if sac.epoch_step % 1 == 0:
                    sac.update_weights()

        if episode % 1 == 0:
            sac.policy.save_weights(args.model_path + args.model_name + '/policy.ckpt')
            sac.q1.save_weights(args.model_path + args.model_name + '/q1.ckpt')
            sac.q2.save_weights(args.model_path + args.model_name + '/q2.ckpt')
            sac.target_q1.save_weights(args.model_path + args.model_name + '/target_q1.ckpt')
            sac.target_q2.save_weights(args.model_path + args.model_name + '/target_q2.ckpt')
            with open('alpha_value.txt', 'w') as f:
                f.write(str(float(sac.alpha.value())))
            with open('alpha_optimizer.pickle', 'wb') as f:
                pickle.dump(sac.alpha_optimizer, f)


        episode_rewards.append(episode_reward)
        final_metrics.append(current_metric)
        episode += 1
        avg_episode_reward = sum(episode_rewards[-100:]) / len(episode_rewards[-100:])
        avg_final_metric = sum(final_metrics[-100:]) / len(final_metrics[-100:])

        print(f"Episode {episode} reward: {episode_reward}")
        print(f"{episode} Average episode reward: {avg_episode_reward}")
        print(f'Episode {episode} final value of metric {current_metric}')
        print(f'{episode} Average final value of metric {avg_final_metric}')
        with writer.as_default():
            tf.summary.scalar("episode_reward", episode_reward, episode)
            tf.summary.scalar("avg_episode_reward", avg_episode_reward, episode)