import random
import numpy as np

class NStep:
    def __init__(self, action_space, alpha, gamma, epsilon, n):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n
        self.q_values = {}

        self.reward_store = []
        self.state_store = []
        self.action_store = []

    def sample_action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            action = self.argmax(state)

        self.action_store.append(action)
        return action
        
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
        return self.q_values.get((state, action), 0)
    
    def learn(self, state, action, reward, next_state, next_action):
        pass

    def store_state(self, state):
        self.state_store.append(state)

    def store_reward(self, reward):
        self.reward_store.append(reward)