import os
import numpy as np
import matplotlib.pyplot as plt

from Environments.SimpleEnvs.CliffEnv import CliffEnv
from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv

from agents.q_learning import QLearning
from agents.sarsa import Sarsa
from agents.n_step import NStep


def show(env, current_state, reward=None):
    env.show()
    print(f"Raw state: {current_state}")
    if reward:
        print(f"Reward: {reward}")


def get_action_from_user(valid_actions):
    key = input()
    while key not in valid_actions:
        key = input()
    return valid_actions[key]


def play_simple_env(simple_env):
    key2action = {'a': "left", 'd': "right", 'w': "up", 's': "down"}
    s = simple_env.reset()
    show(simple_env, s)
    done = False
    while not done:
        print("Action: ", end="")
        action = get_action_from_user(key2action)
        s, r, done = simple_env.step(action)
        show(simple_env, s, r)

def plot_returns(returns, filename="returns.png"):
    plt.figure()
    plt.plot(returns)
    plt.xlabel("Episodes")
    plt.ylabel("Average Return")
    plt.ylim(-200, returns.max())
    plt.savefig(os.path.join("imgs", filename))

def run_q_learning(env, num_episodes):
    agent = QLearning(env.action_space)
    episode_returns = np.zeros(num_episodes)
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_return = 0
        while not done:
            action = agent.sample_action(state)
            next_state, reward, done = env.step(action)
            episode_return += reward
            agent.learn(state, action, reward, next_state)
            state = next_state
        episode_returns[episode] = episode_return
    return episode_returns

def run_sarsa(env, num_episodes):
    agent = Sarsa(env.action_space)
    episode_returns = np.zeros(num_episodes)
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_return = 0
        action = agent.sample_action(state)
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.sample_action(next_state)
            episode_return += reward
            agent.learn(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
        episode_returns[episode] = episode_return
    return episode_returns

# TODO: Refactor del año
def run_n_step(env, num_episodes):
    agent = NStep(env.action_space, 4)
    episode_returns = np.zeros(num_episodes)
    for episode in range(num_episodes):
        agent.reset_stores()
        state = env.reset()
        agent.store_state(state)
        action = agent.sample_action(state)
        agent.store_action(action)
        episode_len = float('inf') # T in pseudo code
        t = 0

        episode_return = 0
        while True:
            if t < episode_len:
                next_state, reward, done = env.step(action)
                agent.store_state(next_state)
                agent.store_reward(reward)
                episode_return += reward
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
        episode_returns[episode] = episode_return
    return episode_returns

if __name__ == "__main__":
    env = CliffEnv()
    # env = EscapeRoomEnv()
    num_of_experiments = 100
    num_of_episodes = 500
    function = run_n_step
    average_returns = np.zeros(num_of_episodes)
    for i in range(num_of_experiments):
        returns = function(env, num_of_episodes)
        average_returns += (returns - average_returns) / (i + 1)
    plot_returns(average_returns, filename=f"{'_'.join(function.__name__.split('_')[1:])}.png")
