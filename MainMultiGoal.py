import os
import numpy as np
import matplotlib.pyplot as plt

from Environments.MultiGoalEnvs.RoomEnv import RoomEnv
from MainSimpleEnvs import play_simple_env
from agents.q_learning import QLearning

def play_room_env():
    n_episodes = 10
    for _ in range(n_episodes):
        env = RoomEnv()
        play_simple_env(env)

def plot_average_length(lengths, filename="mutli_goal_lengths.png"):
    plt.figure()
    plt.plot(lengths)
    plt.xlabel("Episodes")
    plt.ylabel("Average Length")
    plt.savefig(os.path.join("imgs", filename))

def run_q_learning(env, num_of_episodes):
    agent = QLearning(env.action_space, gamma=0.99, alpha=0.1, epsilon=0.1)
    episode_lengths = np.zeros(num_of_episodes)
    for episode in range(num_of_episodes):
        state = env.reset()
        done = False
        episode_length = 0
        while not done:
            action = agent.sample_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            episode_length += 1
        episode_lengths[episode] = episode_length
    return episode_lengths


if __name__ == '__main__':
    num_of_experiments = 100
    num_of_episodes = 500
    
    average_returns = np.zeros(num_of_episodes)
    function = run_q_learning
    for i in range(num_of_experiments):
        env = RoomEnv()
        returns = function(env, num_of_episodes)
        average_returns += (returns - average_returns) / (i + 1)
    plot_average_length(average_returns)