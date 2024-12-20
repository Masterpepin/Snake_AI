from snake_env import SnakeEnv
from dqn_agent import DQNAgent
import numpy as np
import os
import pygame
import tensorflow as tf
from dqn_agent import mse
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize environment and agent
env = SnakeEnv()
state_shape = (30, 20, 1)  # Grid dimensions (30x20) with 1 channel
action_size = 4  # Four possible actions: up, down, left, right
agent = DQNAgent(state_shape, action_size)
agent.model = load_model("models/snake_dqn_final.h5", custom_objects={"mse": mse})

# Recreate the optimizer and recompile the model
agent.model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

# Training parameters
episodes = 1000  # Continue training for 10 more games
fps = 10  # Frames per second

# Initialize Pygame clock
clock = pygame.time.Clock()

# Continue training loop
for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, *state_shape])
    total_reward = 0
    done = False

    while not done:
        # Render the game for visualization
        env.render()

        # Agent selects an action
        action = agent.act(state)

        # Environment processes the action
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, *state_shape])

        # Store the experience in replay memory
        agent.remember(state, action, reward, next_state, done)

        # Update the state
        state = next_state
        total_reward += reward

        # Train the agent only every 10 steps
        if episode % 10 == 0:
            agent.replay()

        # Limit frame rate
        clock.tick(fps)

    print(f"Episode: {episode + 1}/{episodes}, Score: {env.score}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # Update the target network periodically
    if episode % 10 == 0:
        agent.update_target_model()

    if episode % 50 == 0:  # Save the model every 50 episodes
        agent.model.save(f"models/snake_dqn_{episode}.h5")


# Quit Pygame after training
env.close()
pygame.quit()
