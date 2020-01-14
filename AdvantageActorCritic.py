import numpy as np
import tensorflow as tf


class ActionValueModel(tf.keras.Model):
    def __init__(self, num_hidden_layer_units, num_different_actions):
        super().__init__('mlp_policy')
        self.hidden_layer_action = tf.keras.layers.Dense(num_hidden_layer_units, activation=tf.keras.activations.relu)
        self.output_layer_action = tf.keras.layers.Dense(num_different_actions)
        self.hidden_layer_value = tf.keras.layers.Dense(num_hidden_layer_units, activation=tf.keras.activations.relu)
        self.output_layer_value = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        input_to_hidden_layers = tf.convert_to_tensor(inputs)
        hidden_layer_action_output = self.hidden_layer_action(input_to_hidden_layers)
        output_layer_action_output = self.output_layer_action(hidden_layer_action_output)
        hidden_layer_value_output = self.hidden_layer_value(input_to_hidden_layers)
        output_layer_value_output = self.output_layer_value(hidden_layer_value_output)
        return output_layer_action_output, output_layer_value_output

    def get_action_and_value(self, observation):
        output_layer_action_output, output_layer_value_output = self.predict_on_batch(np.expand_dims(observation, 0))
        action = np.squeeze(tf.squeeze(tf.random.categorical(output_layer_action_output, 1), axis=-1), axis=-1)
        value = np.squeeze(tf.squeeze(output_layer_value_output, axis=-1), axis=-1)
        return action, value


class A2CAgent:
    def __init__(self, environment, gamma=np.nextafter(1, 0), entropy_factor=1E-4, value_factor=0.5, num_hidden_layer_units=128):
        self.environment = environment
        self.gamma = gamma
        self.entropy_factor = entropy_factor
        self.value_factor = value_factor
        self.model = ActionValueModel(num_hidden_layer_units, self.environment.action_space.n)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=[self.action_loss, self.value_loss])

    def action_loss(self, actions_and_advantages, action_logits):
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        policy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(actions, action_logits, sample_weight=advantages)
        entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(action_logits, action_logits)
        return policy_loss - self.entropy_factor * entropy_loss

    def value_loss(self, values, value_logits):
        return self.value_factor * tf.keras.losses.MeanSquaredError()(values, value_logits)

    def test(self, render=False):
        actions, values, observations, rewards, done = [], [], [self.environment.reset()], [], False
        while not done:
            next_action, next_value = self.model.get_action_and_value(observations[-1])
            actions.append(next_action)
            values.append(next_value)
            current_observation, current_reward, done = self.environment.step(actions[-1])[:-1]
            observations.append(current_observation)
            rewards.append(current_reward)
            if render:
                self.environment.render()
        return np.array(actions), np.array(values), np.array(observations[:-1]), np.array(rewards)

    def get_returns_and_advantages(self, rewards, values):
        returns = np.copy(rewards)
        for i in reversed(range(returns.shape[0] - 1)):
            returns[i] += self.gamma * returns[i + 1]
        advantages = returns - values
        return returns, advantages

    def train(self, history_length=10, log=True):
        episodic_rewards = [0]
        while np.min(episodic_rewards[-history_length:]) < self.environment.spec.max_episode_steps:
            actions, values, observations, rewards = self.test()
            episodic_rewards.append(np.sum(rewards))
            returns, advantages = self.get_returns_and_advantages(rewards, values)
            self.model.train_on_batch(observations, [np.array(list(zip(actions, advantages))), returns])
            if log:
                print(f'episodic reward while training: {episodic_rewards[-1]}')
        return np.array(episodic_rewards)
