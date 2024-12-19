from snake_env import SnakeEnv
from dqn_agent import DQNAgent
import numpy as np
import time

# Initialize environment and agent
env = SnakeEnv()
state_shape = (30, 20, 1)  # Grid dimensions (30x20) with 1 channel
action_size = 4  # Four possible actions: up, down, left, right
agent = DQNAgent(state_shape, action_size)

# Training parameters
episodes = 100  # Number of games the agent will play
max_steps_per_episode = 500  # Maximum steps in a single game
batch_size = 32  # Number of samples to train on at each step

# Training loop
for episode in range(episodes):
    state = env.reset()  # Reset the game
    state = np.reshape(state, [1, *state_shape])  # Reshape for the neural network
    total_reward = 0

    for step in range(max_steps_per_episode):
        env.render()  # Visualize the game
        action = agent.act(state)  # Agent takes an action

        # Environment processes the action
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, *state_shape])  # Reshape for the neural network

        # Store the experience in replay memory
        agent.remember(state, action, reward, next_state, done)

        # Update the state
        state = next_state
        total_reward += reward

        # Train the agent with a batch from memory
        agent.replay()

        if done:  # Game over
            print(f"Episode: {episode + 1}/{episodes}, Score: {env.score}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            break

        time.sleep(0.05)  # Slow down for visualization

    # Update the target network periodically
    if episode % 10 == 0:
        agent.update_target_model()

    # Save the model periodically
    if episode % 50 == 0:
        agent.model.save(f"models/snake_dqn_{episode}.h5")
