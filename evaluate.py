from snake_env import SnakeEnv
from dqn_agent import DQNAgent
import numpy as np
from tensorflow.keras.models import load_model
import pygame
import time

# Initialize environment and agent
env = SnakeEnv()
state_shape = (30, 20, 1)
action_size = 4
agent = DQNAgent(state_shape, action_size)

# Load the trained model
agent.model = load_model("models/snake_dqn_final.h5")

# Play without training
episodes = 5  # Number of games to play
fps = 10  # Frames per second
clock = pygame.time.Clock()

for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, *state_shape])
    total_reward = 0
    done = False

    while not done:
        # Render the game
        env.render()

        # Agent selects an action (no epsilon-greedy here, just exploit)
        action = np.argmax(agent.model.predict(state)[0])

        # Environment processes the action
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, *state_shape])
        state = next_state
        total_reward += reward

        # Control game speed
        clock.tick(fps)

    print(f"Episode {episode + 1}/{episodes}: Total Reward: {total_reward:.2f}")

# Quit Pygame
env.close()
pygame.quit()
