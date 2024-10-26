from collections import defaultdict
import numpy as np
class DynaQ:
    def __init__(self, action_space, alpha=0.1, gamma=0.9, epsilon=0.1, n=5):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n
        self.q_table = defaultdict(lambda: np.zeros(action_space))
        self.model = defaultdict(lambda: (0, 0))

    def sample_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        self.q_table[state][action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action])
        self.model[state] = (reward, next_state)
        for _ in range(self.n):
            state = np.random.choice(list(self.model.keys()))
            action = np.random.choice(self.action_space)
            reward, next_state = self.model[state]
            self.q_table[state][action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action])