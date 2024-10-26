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

        self.reward_store = [0] # Para guardar la primera recompensa obtenida como R_{t+1}
        self.state_store = []
        self.action_store = []

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
        return self.q_values.get((state, action), 0)
    
    def learn(self, tau, episode_len):
        g = sum(self.reward_store[i] * (self.gamma ** (i - tau - 1)) for i in range(tau + 1, min(tau + self.n, episode_len) + 1))
        if tau + self.n < episode_len:
            g += self.get_q_value(self.state_store[tau + self.n], self.action_store[tau + self.n]) * (self.gamma ** self.n)
        self.q_values[(self.state_store[tau], self.action_store[tau])] = self.get_q_value(self.state_store[tau], self.action_store[tau]) + self.alpha * (g - self.get_q_value(self.state_store[tau], self.action_store[tau]))

    def store_state(self, state):
        self.state_store.append(state)

    def store_reward(self, reward):
        self.reward_store.append(reward)

    def store_action(self, action):
        self.action_store.append(action)

    def reset_stores(self):
        self.reward_store = [0]
        self.state_store = []
        self.action_store = []