import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from src.ddpg.ActorCritic import ActorNetwork, CriticNetwork
from src.libs.replay_buffer import ReplayBuffer
from src.libs.rl import AbstractReinforcementLearningModel
from pathlib import Path
import numpy as np


class DDPG(AbstractReinforcementLearningModel):
    def __init__(self, epoch_step=1, alpha=0.001, beta=0.002,
                 gamma=0.99, n_actions=1, tau=0.005, layer_size=256, batch_size=64, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.epoch_step = epoch_step

        self.actor = ActorNetwork(n_actions=n_actions, name='actor', fc1_dims=layer_size, fc2_dims=layer_size)
        self.critic = CriticNetwork(name='critic', fc1_dims=layer_size, fc2_dims=layer_size)
        self.target_actor = ActorNetwork(n_actions=n_actions,
                                         name='target_actor', fc1_dims=layer_size, fc2_dims=layer_size)
        self.target_critic = CriticNetwork(name='target_critic', fc1_dims=layer_size, fc2_dims=layer_size)

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_policy_network(tau=1)

    def update_policy_network(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    def save_model(self, model_folder: Path, model_name: str):
        model_path = model_folder / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        tf.saved_model.save(self.actor, str(model_path / 'a.h5'))
        tf.saved_model.save(self.target_actor, str(model_path / 'ta.h5'))
        tf.saved_model.save(self.critic, str(model_path / 'c.h5'))
        tf.saved_model.save(self.target_critic, str(model_path / 'tc.h5'))

    def load_model(self, model_folder: Path, model_name: str):
        model_path = model_folder / model_name
        self.actor = tf.saved_model.load(str(model_path / 'a.h5'))
        self.target_actor = tf.saved_model.load(str(model_path / 'ta.h5'))
        self.critic = tf.saved_model.load(str(model_path / 'c.h5'))
        self.target_critic = tf.saved_model.load(str(model_path / 'tc.h5'))

    def sample_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float64)
        actions = self.actor(state)
        if actions == np.nan:
            print(actions)
            print('ACTION IS NONE. MODEL DOES NOT CONVERGE!')
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions],
                                        mean=1.0,
                                        stddev=self.noise,
                                        dtype=tf.float64)
        # note that if the env has an action > 1, we have to multiply by
        # max action at some point
        actions = tf.clip_by_value(t=actions, clip_value_min=0, clip_value_max=10)

        return actions[0]

    def train(self, current_states, actions, rewards, next_states, ends):

        current_states = tf.convert_to_tensor(current_states, dtype=tf.float64)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float64)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float64)
        actions = tf.convert_to_tensor(actions, dtype=tf.float64)
        state_action = tf.concat([current_states, actions], axis=1)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            next_state_action = tf.concat([next_states, target_actions], axis=1)
            critic_value_ = tf.squeeze(self.target_critic(next_state_action), 1)
            critic_value = tf.squeeze(self.critic(state_action), 1)
            target = rewards + self.gamma * critic_value_ * (1 - ends)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(current_states)
            new_state_actions = tf.concat([current_states, new_policy_actions], axis=1)
            actor_loss = -self.critic(new_state_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_policy_network()

        return actor_loss, critic_loss

    def complex_training(self, buffer: ReplayBuffer, training_params: dict, verbose: bool = False):
        print('Start training')
        for epoch in range(training_params['epochs']):
            # Randomly sample minibatch of transitions from replay buffer
            current_states, actions, rewards, next_states, ends = buffer.fetch_sample(
                num_samples=training_params['batch_size'])

            # Perform single step of gradient descent on Q and policy network
            critic_loss, actor_loss = self.train(current_states, actions, rewards, next_states, ends)
            if verbose:
                print(epoch, critic_loss.numpy(), actor_loss.numpy())

            self.epoch_step += 1
