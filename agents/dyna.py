from collections import defaultdict
import numpy as np
class DynaQ:
    def __init__(self, action_space, alpha=0.1, gamma=0.9, epsilon=0.1, n=5):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n
        self.q_values = defaultdict(lambda: 0)
        self.model = {} 

    def sample_action(self, state):
        if np.random.random() < self.epsilon:
            return str(np.random.choice(self.action_space))
        return self.argmax(state)
    
    def argmax(self, state):
        max_action = None
        q_max = float('-inf')
        for action in self.action_space:
            q_value = self.q_values[(state, action)]
            if q_value > q_max:
                q_max = q_value
                max_action = [action]
            elif q_value == q_max:
                max_action.append(action)
        return str(np.random.choice(max_action))
    
    
    def learn(self, state, action, reward, next_state):
        #print(type(action))
        self.q_values[(state, action)] += self.alpha * (reward + self.gamma * np.max(self.q_values[(next_state,action)]) - self.q_values[(state,action)])
        self.model[state] = (reward, next_state)
        for _ in range(self.n):
            model_keys = list(self.model.keys())
            state_idx = np.random.choice(len(model_keys))
            state = model_keys[state_idx]
            action = np.random.choice(self.action_space)
            reward, next_state = self.model[state]
            self.q_values[(state,action)] += self.alpha * (reward + self.gamma * np.max(self.q_values[(next_state,action)]) - self.q_values[(state, action)])