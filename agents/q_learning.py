import random
import numpy as np

class QLearning:
    def __init__(self, action_space, gamma=0.9, alpha=0.1, epsilon=0.1, q_baseline=0):
        self.action_space = action_space
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q_values = {}
        self.q_baseline = q_baseline

    def sample_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.action_space)
        return self.argmax(state)
    
    def get_q_value(self, state, action):
        # This method is added to simulate the initialization of the Q-values
        return self.q_values.get((state, action), self.q_baseline)
    
    def argmax(self, state):
        max_action = None
        q_max = float('-inf')
        for action in self.action_space:
            q_value = self.get_q_value(state, action)
            if q_value > q_max:
                q_max = q_value
                max_action = [action]
            elif q_value == q_max:
                max_action.append(action)
        return random.choice(max_action)

    def learn(self, state, action, reward, next_state, done):
        if done:
            td_error = reward - self.get_q_value(state, action)
        else:
            td_error = reward + self.gamma * self.get_q_value(next_state, self.argmax(next_state)) - self.get_q_value(state, action)
        self.q_values[(state, action)] = self.get_q_value(state, action) + self.alpha * td_error