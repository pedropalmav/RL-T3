import os
import numpy as np
import matplotlib.pyplot as plt

from Environments.MultiGoalEnvs.RoomEnv import RoomEnv
from MainSimpleEnvs import play_simple_env
from agents.q_learning import QLearning
from agents.sarsa import Sarsa
from agents.n_step import NStep
from agents.multi_goal.multi_goal_q_learning import MultiGoalQLearning

def play_room_env():
    n_episodes = 10
    for _ in range(n_episodes):
        env = RoomEnv()
        play_simple_env(env)

def plot_average_length(lengths, filename="mutli_goal_lengths.png"):
    plt.figure()
    for key, value in lengths.items():
        plt.plot(value, label=key)
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
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1} finished")
    return episode_lengths

def run_sarsa(env, num_episodes):
    agent = Sarsa(env.action_space, gamma=0.99, alpha=0.1, epsilon=0.1)
    episode_lengths = np.zeros(num_episodes)
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        action = agent.sample_action(state)
        episode_length = 1
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.sample_action(next_state)
            agent.learn(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            episode_length += 1
        episode_lengths[episode] = episode_length
    return episode_lengths

def run_n_step(env, num_episodes):
    agent = NStep(env.action_space, 8, gamma=0.99, alpha=0.1, epsilon=0.1)
    episode_lengths = np.zeros(num_episodes)
    for episode in range(num_episodes):
        agent.reset_stores()
        state = env.reset()
        agent.store_state(state)
        action = agent.sample_action(state)
        agent.store_action(action)
        episode_len = float('inf') # T in pseudo code
        t = 0

        while True:
            if t < episode_len:
                next_state, reward, done = env.step(action)
                agent.store_state(next_state)
                agent.store_reward(reward)
                if done:
                    episode_len = t + 1
                else:
                    action = agent.sample_action(next_state)
                    agent.store_action(action)
            tau = t - agent.n + 1
            if tau >= 0:
                agent.learn(tau, episode_len)
            if tau == episode_len - 1:
                break
            t += 1
        episode_lengths[episode] = episode_len
    return episode_lengths

def run_multi_goal_q_learning(env, num_of_episodes):
    agent = MultiGoalQLearning(env.action_space, gamma=0.99, alpha=0.1, epsilon=0.1, q_baseline=1.0)
    goals = env.goals
    episode_lengths = np.zeros(num_of_episodes)
    for episode in range(num_of_episodes):
        state = env.reset()
        done = False
        episode_length = 0
        while not done:
            action = agent.sample_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, goals)
            state = next_state
            episode_length += 1
        episode_lengths[episode] = episode_length
    return episode_lengths


if __name__ == '__main__':
    num_of_experiments = 100
    num_of_episodes = 500
    
    # functions = [run_q_learning, run_sarsa, run_n_step]
    functions = [run_multi_goal_q_learning]
    agents_avg_lengths = {}
    for function in functions:
        average_lengths = np.zeros(num_of_episodes)
        for i in range(num_of_experiments):
            env = RoomEnv()
            lengths = function(env, num_of_episodes)
            average_lengths += (lengths - average_lengths) / (i + 1)
            print(f"Experiment {i + 1} finished")
        agents_avg_lengths[function.__name__] = average_lengths
    plot_average_length(agents_avg_lengths)