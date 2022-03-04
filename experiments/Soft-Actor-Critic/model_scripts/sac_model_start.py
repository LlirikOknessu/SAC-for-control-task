import argparse
import logging

import numpy as np
import tensorflow as tf
from pathlib import Path

from src.sac.sac import SoftActorCritic
from src.libs.replay_buffer import ReplayBuffer
from src.libs.connector import Connector
from src.libs.utils import reward_gauss


EPSILON = 1e-7
tf.keras.backend.set_floatx('float64')

logging.basicConfig(level='INFO')

parser = argparse.ArgumentParser(description='SAC')
parser.add_argument('--params', '-p', type=str, default='params.yaml', required=True,
                    help='param file path')
parser.add_argument('--model_path', '-mp', type=str, default='data/models/experiment/', required=True,
                    help='path to save model')
parser.add_argument('--model_name', '-mn', type=str, default=f'model_v19', required=True,
                    help='name of the saved model')
parser.add_argument('--model_observation', '-mo', default=5,
                    help='Shape of model observation space')
parser.add_argument('--model_action_space', '-ma', default=1,
                    help='Shape of action space')

parser.add_argument('--seed', type=int, default=42, required=False,
                    help='random seed')
parser.add_argument('--env_name', type=str, default='BallTube-v1', required=False,
                    help='name of the gym environment with version')
parser.add_argument('--render', type=bool, default=False, required=False,
                    help='set gym environment to render display')

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
            next_state, metric, y_target, done = connector_to_model.receive()

            y_true = next_state[1]
            reward = reward_gauss(target_value=y_target, true_value=y_true, scale=1)
            episode_reward += reward

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
            print('Start training')
            for epoch in range(args.epochs):

                # Randomly sample minibatch of transitions from replay buffer
                current_states, actions, rewards, next_states, ends = replay.fetch_sample(num_samples=args.batch_size)

                # Perform single step of gradient descent on Q and policy network
                critic1_loss, critic2_loss, actor_loss, alpha_loss = sac.train(current_states, actions, rewards,
                                                                               next_states, ends)
                if args.verbose:
                    print(episode, global_step, epoch, critic1_loss.numpy(),
                          critic2_loss.numpy(), actor_loss.numpy(), episode_reward)

                sac.epoch_step += 1

        if episode % 5 == 0:
            sac.save_model(Path(args.model_path), args.model_name)

        episode_rewards.append(episode_reward)
        episode += 1
        avg_episode_reward = sum(episode_rewards[-100:]) / len(episode_rewards[-100:])

        print(f"Episode {episode} reward: {episode_reward / 1}")
        print(f"{episode} Average episode reward: {avg_episode_reward / 1}")
        print(f'Final value of metric {current_metric}')
        print(f'Target {y_target}')
