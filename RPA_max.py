import pygame
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import random
import time
import openai_vision as eye


# For text embedding

max_length = 10

# Initialize Pygame and create a screen
pygame.init()
width, height = 400, 400
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

# Neural Network and episode counter setup
model_path = 'model.h5'
success_path = 'successes.txt'
attempts_path = 'attempts.txt'
epsilon_path = 'epsilon.txt'



def load_file(counter_file, default_value=None):
    try:
        with open(counter_file, 'r') as file:
            return file.read()
        

    except FileNotFoundError:


        if default_value: 
            return default_value
        else: 
            return 0  # If the file does not exist, use the default starting value

def save_file(counter_file, counter_value):
    with open(counter_file, 'w') as file:
        file.write(str(counter_value))

def initialize_model(model_path):
    """
    Initialize the neural network model.

    :param model_path: Path to the saved model.
    :return: The loaded or newly created model.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except IOError:
        input_shape = (2 + max_length,)  # Adjust based on state size and max_length
        model = Sequential([
            Dense(64, activation='relu', input_shape=input_shape),
            Dense(32, activation='relu'),
            Dense(4, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Reset the success and attempts counters by writing zero to their files
        successes = 0
        attempts = 0
        save_file(success_path, successes)
        save_file(attempts_path, attempts)
        save_file(epsilon_path, 1.0)
        print("Counters reset.")
        
        print("New model created.")
        return model


# Define a function to get the state
def get_state(target, agent):
    return np.array([target[0] - agent[0], target[1] - agent[1]])

# Define a function to train the model
def train_model(state, action, reward, next_state, done, model):
    target = reward
    if not done:
        target = reward + 0.99 * np.amax(model.predict(next_state.reshape(1, -1))[0])
    target_f = model.predict(state.reshape(1, -1))
    target_f[0][action] = target
    model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

def process_description(description, state, vocab_size=10000, max_length=10):
    """
    Process the description to be used in the model.

    :param description: The description string.
    :param state: The current state of the agent.
    :param vocab_size: Size of the vocabulary for encoding.
    :param max_length: Maximum length of the encoded vector.
    :return: Combined state and encoded description.
    """
    # Encode the description using one-hot encoding
    encoded = one_hot(description, vocab_size)
    # Pad the encoded description
    padded = pad_sequences([encoded], maxlen=max_length, padding='post')
    # Flatten the padded description
    flattened = padded.flatten()

    # Append the flattened description to the state
    extended_state = np.append(state, flattened)

    return extended_state


def choose_action_and_update_position(state, agent, epsilon, model):
    """
    Choose an action based on ε-greedy strategy and update the agent's position.

    :param epsilon: Exploration rate.
    """
    step_size = 5  # Define the step size for each movement
    action_taken = True

    if np.random.rand() <= epsilon:
        # Take a random action
        action = np.random.randint(0, 5)
    else:
        # Take the best action according to the model
        action = np.argmax(model.predict(state.reshape(1, -1)))

    # Update agent's position based on the chosen action
    if action == 0:  # Up
        new_agent_pos = agent[0], agent[1] - step_size
    elif action == 1:  # Down
        new_agent_pos = agent[0], agent[1] + step_size
    elif action == 2:  # Left
        new_agent_pos = agent[0] - step_size, agent[1]
    elif action == 3:  # Right
        new_agent_pos = agent[0] + step_size, agent[1]
    # Incorporate the visual analysis into the decision-making process
    elif action == 4:  # If the action is to see
        success, description = eye.see_computer_screen()
        if success:
            # Process the description to influence the decision
            # For example, modify the state with the new information
            state_with_vision_info = process_description(description, state)
            # Use the updated state for the next decision
            action = np.argmax(model.predict(state_with_vision_info.reshape(1, -1)))
        else:
            # Handle failure to get a description
            # You could choose a default action or use the original state
            action = np.argmax(model.predict(state.reshape(1, -1)))
    else:
        new_agent_pos = agent
        action_taken = False

    return action, new_agent_pos, action_taken





def calculate_reward(target, agent, time_remaining, success_threshold, width, successes, action_taken):
    """
    Calculate the reward for the agent's current state, and update success and done status.
    
    :param target: Tuple of (x, y) coordinates for the target.
    :param agent: Tuple of (x, y) coordinates for the agent.
    :param time_remaining: Time remaining for the agent to reach the target.
    :param success_threshold: Distance threshold for considering the agent has reached the target.
    :param width: Width of the screen, used for scaling the distance-based reward.
    :param successes: The current count of successes.
    :return: A tuple containing the numerical reward value, updated success counter, and done status.
    Additional parameter:
    :param action_taken: Boolean indicating if an action was taken.
    """
    distance = np.linalg.norm(np.array(target) - np.array(agent))
    done = False
    inaction_penalty = -0.1  # Define the penalty for inaction
    distance_penalty_scale = 1.0  # Increase this value to make distance more impactful

    if distance < success_threshold:
        reward = 1
        successes += 1
        done = True
    elif time_remaining <= 0:
        reward = -1
        done = True
    else:
        # Apply a more aggressive penalty based on the distance
        reward = -distance_penalty_scale * distance / width
        if not action_taken:
            reward += inaction_penalty  # Apply penalty for inaction

    return reward, successes, done

def main():

    model = initialize_model(model_path)

    # Main loop
    running = True
    time_limit = 5  # Time limit of 5 seconds
    successes = int(load_file(success_path, 0))  # Load the number of successes
    attempts = int(load_file(attempts_path, 0))  # Load the total number of attempts
    epsilon = float(load_file(epsilon_path, default_value=1.0)) # Load ε at the start

    # Initialize ε parameters
    epsilon_min = 0.01  # Minimum value of ε
    epsilon_decay = 0.995  # Decay rate of ε per episode


    while running:
        attempts += 1  # Increment attempts counter

        # Initialize the start position and time
        agent = (random.randint(0, width), random.randint(0, height))
        start_time = time.time()
        
        # Initialize the state
        target = (width // 2, height // 2)
        state = get_state(target, agent)
        
        # Run episode
        done = False
        while not done:
            screen.fill((0, 0, 0))
            pygame.draw.circle(screen, (255, 255, 255), target, 10)
            pygame.draw.circle(screen, (255, 0, 0), agent, 10)

            # Display the life iteration counter
            life_text = font.render(f"Life: {attempts}", True, (255, 255, 255))
            screen.blit(life_text, (10, 10))

            # Update and render timer
            elapsed_time = time.time() - start_time
            remaining_time = max(time_limit - elapsed_time, 0)
            timer_text = font.render(f"{remaining_time:.2f}s", True, (255, 255, 255))
            screen.blit(timer_text, (width - 100, 10))

            
            # Update agent's position based on chosen action
            action, agent, action_taken = choose_action_and_update_position(state, agent, epsilon, model)

            next_state = get_state(target, agent)

            # Calculate reward and check if the episode is done
            reward, successes, done = calculate_reward(target, agent, remaining_time, 10, width, successes, action_taken)



            # Display the success ratio
            success_ratio = successes / attempts if attempts > 0 else 0
            success_ratio_text = font.render(f"Success ratio: {successes}/{attempts} = {success_ratio:.2f}", True, (255, 255, 255))
            screen.blit(success_ratio_text, (10, height - 30))

            pygame.display.flip()
            clock.tick(60)
            
            # Process Pygame events
            for event in pygame.event.get():
                if event.type is pygame.QUIT:
                    running = False
                    done = True
            
            # If the episode is done (either success or failure), train the model
            if done:
                train_model(state, action, reward, next_state, done, model)

        
        # Update ε after each episode
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        # Save the model and counters at the end of each episode
        print('Model Saved')
        model.save(model_path)
        save_file(success_path, successes)
        save_file(attempts_path, attempts)
        save_file(epsilon_path, epsilon)

    # Quit Pygame
    pygame.quit()
