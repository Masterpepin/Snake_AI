from snake_env import SnakeEnv
from dqn_agent import DQNAgent
import numpy as np
import time
import pygame
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize environment and agent
env = SnakeEnv()
state_shape = (30, 20, 1)  # Grid dimensions (30x20) with 1 channel
action_size = 4  # Four possible actions: up, down, left, right
agent = DQNAgent(state_shape, action_size)

# Training parameters
episodes = 10  # Number of games the agent will play
max_steps_per_episode = 500  # Maximum steps in a single game
render_frequency = 1  # Render every N episodes
fps = 10  # Frames per second (controls snake speed)

# Initialize Pygame clock for consistent timing
clock = pygame.time.Clock()

# Training loop
for episode in range(episodes):
    state = env.reset()  # Reset the game
    state = np.reshape(state, [1, *state_shape])  # Reshape for the neural network
    total_reward = 0
    done = False
    print(f"Starting Episode {episode + 1}/{episodes}")

    # Timer for consistent frame rate
    last_time = time.time()

    # Add a counter for training steps
    # Add a counter for training steps
    training_steps = 0

    while not done:
        # Render the game for visualization
        if episode % render_frequency == 0:
            env.render()

        # Agent selects an action
        action = agent.act(state)

        # Environment processes the action
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, *state_shape])  # Reshape for the neural network

        # Store the experience in replay memory
        agent.remember(state, action, reward, next_state, done)

        # Update the state
        state = next_state
        total_reward += reward

        # Train the agent only every 10 steps
        training_steps += 1
        if training_steps % 10 == 0:
            agent.replay()

        # Limit the frame rate
        clock.tick(fps)

    # Log episode results
    print(f"Episode: {episode + 1}/{episodes}, Score: {env.score}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # Update the target network periodically
    if episode % 10 == 0:
        agent.update_target_model()

    if episode % 50 == 0:  # Save the model every 50 episodes
        agent.model.save(f"models/snake_dqn_{episode}.h5")

# Quit Pygame after training
agent.model.save("models/snake_dqn_final.h5")
env.close()
pygame.quit()

