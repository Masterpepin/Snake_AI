
import pygame
import time
import random

# Initialize Pygame
pygame.init()

# Screen dimensions (30x20 blocks of 20px each = 600px)
BLOCK_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 20
WIDTH, HEIGHT = GRID_WIDTH * BLOCK_SIZE, GRID_HEIGHT * BLOCK_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 125, 0)  # Head color
GRAY = (128, 128, 128)  # Border color

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

# Clock
clock = pygame.time.Clock()

# Game over message
font_style = pygame.font.SysFont(None, 50)

def message(msg, color):
    text = font_style.render(msg, True, color)
    screen.blit(text, [WIDTH / 6, HEIGHT / 3])

def game_loop():
    game_over = False
    game_close = False

    x1, y1 = WIDTH // 2, HEIGHT // 2
    x1_change, y1_change = 0, 0

    direction = None  # Current direction of the snake
    snake_list = []
    length_of_snake = 1

    # Food position
    food_x = round(random.randrange(1, GRID_WIDTH - 1) * BLOCK_SIZE)
    food_y = round(random.randrange(1, GRID_HEIGHT - 1) * BLOCK_SIZE)

    while not game_over:
        while game_close:
            screen.fill(BLACK)
            message("Game Over! Press Q to Quit or C to Play Again", RED)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        game_loop()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and direction != "RIGHT":
                    x1_change, y1_change = -BLOCK_SIZE, 0
                    direction = "LEFT"
                elif event.key == pygame.K_RIGHT and direction != "LEFT":
                    x1_change, y1_change = BLOCK_SIZE, 0
                    direction = "RIGHT"
                elif event.key == pygame.K_UP and direction != "DOWN":
                    x1_change, y1_change = 0, -BLOCK_SIZE
                    direction = "UP"
                elif event.key == pygame.K_DOWN and direction != "UP":
                    x1_change, y1_change = 0, BLOCK_SIZE
                    direction = "DOWN"

        # Check for collision with borders
        if x1 < BLOCK_SIZE or x1 >= WIDTH - BLOCK_SIZE or y1 < BLOCK_SIZE or y1 >= HEIGHT - BLOCK_SIZE:
            game_close = True

        x1 += x1_change
        y1 += y1_change
        screen.fill(BLACK)

        # Draw borders
        pygame.draw.rect(screen, GRAY, [0, 0, WIDTH, BLOCK_SIZE])  # Top border
        pygame.draw.rect(screen, GRAY, [0, 0, BLOCK_SIZE, HEIGHT])  # Left border
        pygame.draw.rect(screen, GRAY, [0, HEIGHT - BLOCK_SIZE, WIDTH, BLOCK_SIZE])  # Bottom border
        pygame.draw.rect(screen, GRAY, [WIDTH - BLOCK_SIZE, 0, BLOCK_SIZE, HEIGHT])  # Right border

        pygame.draw.rect(screen, GREEN, [food_x, food_y, BLOCK_SIZE, BLOCK_SIZE])
        snake_head = [x1, y1]
        snake_list.append(snake_head)
        if len(snake_list) > length_of_snake:
            del snake_list[0]

        for block in snake_list[:-1]:
            if block == snake_head:
                game_close = True

        # Draw the snake
        for index, block in enumerate(snake_list):
            if index == len(snake_list) - 1:  # Head of the snake
                pygame.draw.rect(screen, YELLOW, [block[0], block[1], BLOCK_SIZE, BLOCK_SIZE])
            else:
                pygame.draw.rect(screen, BLUE, [block[0], block[1], BLOCK_SIZE, BLOCK_SIZE])

        pygame.display.update()

        if x1 == food_x and y1 == food_y:
            food_x = round(random.randrange(1, GRID_WIDTH - 1) * BLOCK_SIZE)
            food_y = round(random.randrange(1, GRID_HEIGHT - 1) * BLOCK_SIZE)
            length_of_snake += 1

        clock.tick(10)

    pygame.quit()
    quit()

game_loop()
