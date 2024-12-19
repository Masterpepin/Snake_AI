import numpy as np
import pygame
import random
from gym import Env
from gym.spaces import Discrete, Box

class SnakeEnv(Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()
        # Action space: 4 possible directions (0: Left, 1: Right, 2: Up, 3: Down)
        self.action_space = Discrete(4)
        # Observation space: Grid dimensions and other game-related features
        self.observation_space = Box(low=0, high=255, shape=(30, 20, 1), dtype=np.uint8)

        # Screen dimensions
        self.BLOCK_SIZE = 20
        self.GRID_WIDTH = 30
        self.GRID_HEIGHT = 20
        self.WIDTH, self.HEIGHT = self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE

        # Initialize game state
        self.reset()

    def reset(self):
        # Reset snake position and length
        self.snake_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        self.snake_body = [self.snake_pos[:]]
        self.food_pos = self._place_food()
        self.score = 0
        self.done = False
        self.direction = None
        return self._get_state()

    def step(self, action):
        if action == 0 and self.direction != "RIGHT":
            self.direction = "LEFT"
        elif action == 1 and self.direction != "LEFT":
            self.direction = "RIGHT"
        elif action == 2 and self.direction != "DOWN":
            self.direction = "UP"
        elif action == 3 and self.direction != "UP":
            self.direction = "DOWN"

        # Move the snake
        if self.direction == "LEFT":
            self.snake_pos[0] -= self.BLOCK_SIZE
        elif self.direction == "RIGHT":
            self.snake_pos[0] += self.BLOCK_SIZE
        elif self.direction == "UP":
            self.snake_pos[1] -= self.BLOCK_SIZE
        elif self.direction == "DOWN":
            self.snake_pos[1] += self.BLOCK_SIZE

        # Check for collisions
        if (self.snake_pos in self.snake_body[:-1] or
                self.snake_pos[0] < 0 or self.snake_pos[0] >= self.WIDTH or
                self.snake_pos[1] < 0 or self.snake_pos[1] >= self.HEIGHT):
            self.done = True

        # Add new position to the snake's body
        self.snake_body.append(self.snake_pos[:])
        if self.snake_pos == self.food_pos:
            self.score += 1
            self.food_pos = self._place_food()
        else:
            self.snake_body.pop(0)

        # Calculate reward
        reward = 1 if self.snake_pos == self.food_pos else -0.1
        if self.done:
            reward = -10

        return self._get_state(), reward, self.done, {}

    def render(self, mode="human"):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Snake AI")
        self.screen.fill((0, 0, 0))

        # Draw the food
        pygame.draw.rect(self.screen, (0, 255, 0), (*self.food_pos, self.BLOCK_SIZE, self.BLOCK_SIZE))
        # Draw the snake
        for block in self.snake_body:
            pygame.draw.rect(self.screen, (0, 0, 255), (*block, self.BLOCK_SIZE, self.BLOCK_SIZE))

        pygame.display.flip()

    def _place_food(self):
        x = random.randint(1, self.GRID_WIDTH - 2) * self.BLOCK_SIZE
        y = random.randint(1, self.GRID_HEIGHT - 2) * self.BLOCK_SIZE
        return [x, y]

    def _get_state(self):
        # Use the snake's position, direction, and food position for the state
        return np.array(self.snake_body, dtype=np.uint8).reshape((30, 20, 1))
