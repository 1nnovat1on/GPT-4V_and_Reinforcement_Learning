import pygame
import numpy as np
import tensorflow as tf
import random
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Pygame setup
pygame.init()
width, height = 400, 400
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Neural Network setup
model = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),  # Input: [x, y] relative positions
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')  # Output: Up, Down, Left, Right
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Function to get the state (distance between dots)
def get_state(target, agent):
    return np.array([target[0] - agent[0], target[1] - agent[1]])

# Main loop
running = True
while running:
    screen.fill((0, 0, 0))

    # Place white dot in the center
    target = (width // 2, height // 2)
    pygame.draw.circle(screen, (255, 255, 255), target, 10)

    # Place red dot randomly
    agent = (random.randint(0, width), random.randint(0, height))
    pygame.draw.circle(screen, (255, 0, 0), agent, 10)

    start_time = time.time()
    while True:
        state = get_state(target, agent)
        action = np.argmax(model.predict(state.reshape(1, -1)))

        # Move the red dot based on the action
        if action == 0:  # Up
            agent = (agent[0], agent[1] - 5)
        elif action == 1:  # Down
            agent = (agent[0], agent[1] + 5)
        elif action == 2:  # Left
            agent = (agent[0] - 5, agent[1])
        elif action == 3:  # Right
            agent = (agent[0] + 5, agent[1])

        # Check for time limit
        if time.time() - start_time > 5:
            reward = -1  # Negative feedback
            break

        # Update screen
        screen.fill((0, 0, 0))
        pygame.draw.circle(screen, (255, 255, 255), target, 10)
        pygame.draw.circle(screen, (255, 0, 0), agent, 10)
        pygame.display.flip()
        clock.tick(60)

        # Reward system
        distance = np.linalg.norm(np.array(target) - np.array(agent))
        if distance < 10:
            reward = 1  # Positive feedback
            break
        else:
            reward = -distance / width  # Closer = less negative

        # Neural network training (simplified)
        target_f = model.predict(state.reshape(1, -1))
        target_f[0][action] = reward
        model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
