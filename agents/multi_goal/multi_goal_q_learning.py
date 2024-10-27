from agents.q_learning import QLearning

class MultiGoalQLearning(QLearning):
    def learn(self, state, action, reward, next_state, goals):
        for goal in goals:
            multi_goal_state = (state[0], goal)
            next_multi_goal_state = (next_state[0], goal)
            if state[0] == goal:
                td_error = 1 - self.get_q_value(multi_goal_state, action)
            else:
                td_error = self.gamma * self.get_q_value(next_multi_goal_state, self.argmax(next_multi_goal_state)) - self.get_q_value(multi_goal_state, action)
            self.q_values[(multi_goal_state, action)] = self.get_q_value(multi_goal_state, action) + self.alpha * td_error