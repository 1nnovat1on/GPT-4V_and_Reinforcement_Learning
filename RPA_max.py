import pygame
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences
import random
import time
import openai_vision as eye
import asyncio
import queue

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

# shared resources (Public)
vision_data_queue = queue.Queue()
time_limit = 5

# For text embedding
max_length = 10

# Initialize ε (epsilon) parameters
epsilon_min = 0.01  # Minimum value of ε
epsilon_decay = 0.995  # Decay rate of ε per episode





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
            Dense(5, activation='softmax')
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

def get_state(target, agent, default_vision_data_length=10):
    basic_state = np.array([target[0] - agent[0], target[1] - agent[1]])
    # Initialize vision data with zeros
    vision_data = np.zeros(default_vision_data_length)
    extended_state = np.concatenate((basic_state, vision_data))
    return extended_state


# Define a function to train the model
def train_model(state, action, reward, next_state, done, model):
    target = reward
    if not done:
        target = reward + 0.99 * np.amax(model.predict(next_state.reshape(1, -1))[0])
    target_f = model.predict(state.reshape(1, -1))
    target_f[0][action] = target
    model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

def train_model_minibatch(states, actions, rewards, next_states, dones, model):
  """
  Trains the neural network model on a minibatch of states, actions, rewards, 
  next states, and done flags.

  Args:
    states: A list of current states.
    actions: A list of chosen actions.
    rewards: A list of received rewards.
    next_states: A list of next states.
    dones: A list of done flags indicating episode completion.
    model: The neural network model to be trained.

  Returns:
    None
  """

  # Define minibatch size
  minibatch_size = 32  # Adjust this based on memory constraints and model size

  # Loop through batches in the buffer
  for i in range(0, len(states), minibatch_size):
    # Extract minibatch slices
    batch_states = states[i:i+minibatch_size]
    batch_actions = actions[i:i+minibatch_size]
    batch_rewards = rewards[i:i+minibatch_size]
    batch_next_states = next_states[i:i+minibatch_size]
    batch_dones = dones[i:i+minibatch_size]

    # Calculate target values for each state in the minibatch
    batch_targets = []
    for j in range(minibatch_size):
      if not batch_dones[j]:
        batch_targets.append(batch_rewards[j] + 0.99 * np.amax(model.predict(batch_next_states[j].reshape(1, -1))[0]))
      else:
        batch_targets.append(batch_rewards[j])

    # Convert targets to a NumPy array
    batch_targets = np.array(batch_targets)

    # Calculate and update the model parameters
    target_f = model.predict(batch_states)
    target_f[np.arange(minibatch_size), batch_actions] = batch_targets
    model.fit(batch_states, target_f, epochs=1, verbose=0)



def process_description(description, basic_state, vocab_size=10000, max_length=10):
    # Encode the description using one-hot encoding
    encoded = one_hot(description, vocab_size)
    # Pad the encoded description
    padded = pad_sequences([encoded], maxlen=max_length, padding='post')
    # Flatten the padded description
    vision_data = padded.flatten()

    # Combine basic state with vision data
    extended_state = np.concatenate((basic_state[:2], vision_data))

    return extended_state

def extract_description(response):
    """
    Extract the descriptive text about the image from the response.

    :param response: The response dictionary from see_computer_screen.
    :return: The extracted description as a string.
    """
    try:
        # Access the 'content' field of the 'message' in the first 'choices' element
        description = response['choices'][0]['message']['content']
        return description.strip()  # Remove any leading/trailing whitespace
    except (KeyError, IndexError, TypeError):
        # Return a default message or handle the error as appropriate
        return "Description not available."
    
async def async_screen_reading():
    while True:
        # Asynchronous screen reading logic
        success, description = await eye.see_computer_screen_async()  # This should be an async call
        if success:
            vision_data_queue.put_nowait((description, time.time()))  # Non-blocking put
        await asyncio.sleep(0.1)  # Adjust frequency as needed

async def fetch_vision_data(shared_state):
    while True:
        success, response = await eye.see_computer_screen_async()
        if success:
            shared_state['data'] = extract_description(response)
            shared_state['updated'] = True
        await asyncio.sleep(0.1)  # Adjust as needed

def update_agent_position(agent, action, step_size):
    if action == 0:  # Up
        return agent[0], agent[1] - step_size
    elif action == 1:  # Down
        return agent[0], agent[1] + step_size
    elif action == 2:  # Left
        return agent[0] - step_size, agent[1]
    elif action == 3:  # Right
        return agent[0] + step_size, agent[1]
    else:
        return agent  # No change in position if action is "seeing" or any other undefined action

async def choose_action_and_update_position(state, agent, epsilon, model, shared_state):
    step_size = 5
    action_taken = True

    basic_state = np.array([state[0] - agent[0], state[1] - agent[1]])

    if np.random.rand() <= epsilon:
        action = np.random.randint(0, 5)
    else:
        extended_state = np.concatenate((basic_state, np.zeros(10)))
        extended_state = extended_state.reshape(1, -1)
        action = np.argmax(model.predict(extended_state))

    if action == 4:  # "Seeing" action
        if shared_state.get('updated', False):
            description = shared_state['data']
            state_with_vision_info = process_description(description, basic_state[:2])
            state_with_vision_info = state_with_vision_info.reshape(1, -1)
            action = np.argmax(model.predict(state_with_vision_info))
            shared_state['updated'] = False
        else:
            action = np.argmax(model.predict(extended_state))  # Fallback
    else:
        new_agent_pos = update_agent_position(agent, action, step_size)
        return action, new_agent_pos, action_taken

    new_agent_pos = update_agent_position(agent, action, step_size)
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

async def main():
    
    model = initialize_model(model_path)

    # Main loop
    running = True
    successes = int(load_file(success_path, 0))  # Load the number of successes
    attempts = int(load_file(attempts_path, 0))  # Load the total number of attempts
    epsilon = float(load_file(epsilon_path, default_value=1.0)) # Load ε at the start


    # Start the async screen reading task
    #asyncio.create_task(async_screen_reading())

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
            # action, agent, action_taken = choose_action_and_update_position(state, agent, epsilon, model)
            action, agent, action_taken = await choose_action_and_update_position(state, agent, epsilon, model, vision_data_queue)
            
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

asyncio.run(main())