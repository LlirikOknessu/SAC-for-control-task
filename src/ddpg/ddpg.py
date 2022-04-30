import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from src.ddpg.ActorCritic import ActorNetwork, CriticNetwork
from src.libs.replay_buffer import ReplayBuffer
from src.libs.rl import AbstractReinforcementLearningModel


class DDPG(AbstractReinforcementLearningModel):
    def __init__(self, input_dims, epoch_step=1, alpha=0.001, beta=0.002, env=None,
                 gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 fc1=400, fc2=300, batch_size=64, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.epoch_step = epoch_step

        self.actor = ActorNetwork(n_actions=n_actions, name='actor', fc1_dims=400, fc2_dims=300)
        self.critic = CriticNetwork(name='critic', fc1_dims=400, fc2_dims=300)
        self.target_actor = ActorNetwork(n_actions=n_actions,
                                         name='target_actor', fc1_dims=400, fc2_dims=300)
        self.target_critic = CriticNetwork(name='target_critic', fc1_dims=400, fc2_dims=300)

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

    def save_model(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_model(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def sample_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions],
                                        mean=0.0, stddev=self.noise)
        # note that if the env has an action > 1, we have to multiply by
        # max action at some point
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]

    def train(self, current_states, actions, rewards, next_states, ends,):

        current_states = tf.convert_to_tensor(current_states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            critic_value_ = tf.squeeze(self.target_critic(
                next_states, target_actions), 1)
            critic_value = tf.squeeze(self.critic(current_states, actions), 1)
            target = rewards + self.gamma * critic_value_ * (1 - ends)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(current_states)
            actor_loss = -self.critic(current_states, new_policy_actions)
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
