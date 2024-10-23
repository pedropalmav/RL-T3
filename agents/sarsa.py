import random
import numpy as np

class Sarsa:
    def __init__(self, action_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def sample_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        else:
            return self.argmax(state)
        
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
    
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0)
        
    def learn(self, state, action, reward, next_state, next_action):
        td_error = reward + self.gamma * self.get_q_value(next_state, next_action) - self.get_q_value(state, action)
        self.q_table[(state, action)] = self.get_q_value(state, action) + self.alpha * td_error