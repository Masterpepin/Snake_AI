import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
def mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape  # Shape of the game state (e.g., grid dimensions)
        self.action_size = action_size  # Number of possible actions (4: up, down, left, right)
        self.memory = []  # Replay buffer to store experiences
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995  # Decay factor for exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.learning_rate = 0.001  # Learning rate for the neural network
        self.batch_size = 32  # Size of the training batch
        self.model = self._build_model()  # Neural network for Q-value approximation
        self.target_model = self._build_model()  # Target network for stability
        self.update_target_model()  # Sync target model with main model

    def load_model(self, model_path):
        """Load a pre-trained model."""
        self.model = load_model(model_path)
    def _build_model(self):
        """Builds the Q-network using Keras."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.state_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_size, activation='linear')  # Output layer
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        """Copies the weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Stores experiences in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Selects an action based on the epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore: random action
        q_values = self.model.predict(state)  # Exploit: use the Q-network
        return np.argmax(q_values[0])  # Action with the highest Q-value

    def replay(self):
        """Trains the Q-network using a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return  # Not enough experiences to train
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

