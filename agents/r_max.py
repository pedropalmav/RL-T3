from collections import defaultdict
import numpy as np

class RMax:
    def __init__(self, action_space, R_max, k, gamma, threshold=1e-5, max_iterations=100):
       
        self.action_space = action_space
        self.num_actions = len(action_space)
        self.R_max = R_max  
        self.k = k 
        self.gamma = gamma  

        self.model = defaultdict(lambda: {'reward': R_max, 'transitions': defaultdict(float), 'count': 0})
        self.values = defaultdict(lambda: R_max)  
        self.policy = defaultdict(lambda: 'up')  
        
        self.threshold = threshold
        self.max_iterations = max_iterations

    def learn(self, state, action, reward, next_state):
        self.model[(state, action)]['count'] += 1
        count = self.model[(state, action)]['count']
        
        current_reward = self.model[(state, action)]['reward']
        self.model[(state, action)]['reward'] = ((count - 1) * current_reward + reward) / count
        self.model[(state, action)]['transitions'][next_state] += 1.0 / count

    def value_iteration(self):
        """Run value iteration based on the current model to update the value function and policy."""
        for iteration in range(self.max_iterations):
            
            delta = 0
            new_values = defaultdict(float, self.values)  

            model_keys = list(self.model.keys())
            for (state, _) in model_keys:
                action_values = []
                for action in self.action_space:
                    #print(f'Action: {action}')
                    if self.model[(state, action)]['count'] >= self.k:
                        expected_reward = self.model[(state, action)]['reward']
                        transition_probs = self.model[(state, action)]['transitions']
                        expected_value = sum(transition_probs[s_next] * self.values[s_next] for s_next in transition_probs)
                    else:
                        expected_reward = self.R_max
                        expected_value = self.values[state]
                    
                    action_value = expected_reward + self.gamma * expected_value
                    action_values.append(action_value)

                new_values[state] = max(action_values)
                self.policy[state] = self.action_space[np.argmax(action_values)]

                delta = max(delta, abs(new_values[state] - self.values[state]))

            self.values = new_values
            if delta < self.threshold:
                #print('convergence reached')
                break

    def sample_action(self, state):
        """Return the action based on the current policy for a given state."""
        self.value_iteration()
        action = self.policy[state]
        return action

