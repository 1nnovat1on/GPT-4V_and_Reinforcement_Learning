import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class GameModel:
    def __init__(self, model_path, epsilon_decay_enabled=True, epsilon=1.0, epsilon_min=0.01, epsilon_decay_rate=0.995):
        self.model_path = model_path
        self.epsilon_decay_enabled = epsilon_decay_enabled
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay_rate
        self.model = self.load_model()

    def load_model(self):
        try:
            model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully.")
        except IOError:
            model = Sequential([
                Dense(64, activation='relu', input_shape=(2,)),
                Dense(32, activation='relu'),
                Dense(4, activation='softmax')
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            print("New model created.")
        return model

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + 0.99 * np.amax(self.model.predict(next_state.reshape(1, -1))[0])
        target_f = self.model.predict(state.reshape(1, -1))
        target_f[0][action] = target
        self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

    def save(self):
        self.model.save(self.model_path)
        print('Model saved.')

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, 4)
        else:
            action = np.argmax(self.model.predict(state.reshape(1, -1)))
        return action

    def update_epsilon(self):
        if self.epsilon_decay_enabled:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)
