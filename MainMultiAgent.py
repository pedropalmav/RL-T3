import os
import numpy as np
import matplotlib.pyplot as plt

from Environments.MultiAgentEnvs.HunterAndPreyEnv import HunterAndPreyEnv
from Environments.MultiAgentEnvs.CentralizedHunterEnv import CentralizedHunterEnv
from Environments.MultiAgentEnvs.HunterEnv import HunterEnv
from Environments.MultiAgentEnvs.HunterAndPreyEnv import HunterAndPreyEnv
from MainSimpleEnvs import show, get_action_from_user

from agents.q_learning import QLearning


def play_hunter_env():
    hunter_env = HunterAndPreyEnv()

    key2action = {'a': "left", 'd': "right", 'w': "up", 's': "down", '': "None"}
    num_of_agents = hunter_env.num_of_agents
    s = hunter_env.reset()
    show(hunter_env, s)
    done = False
    while not done:
        print("Hunter A: ", end="")
        hunter1 = get_action_from_user(key2action)
        print("Hunter B: ", end="")
        hunter2 = get_action_from_user(key2action)
        action = hunter1, hunter2
        if num_of_agents == 3:
            print("Prey: ", end="")
            prey = get_action_from_user(key2action)
            action = hunter1, hunter2, prey
        s, r, done = hunter_env.step(action)
        show(hunter_env, s, r)

def plot_average_length(lengths, filename):
    episodes = [i * 100 for i in range(len(lengths))]
    episodes[0] = 1
    plt.figure()
    plt.plot(episodes, lengths)
    plt.xlabel("Episodes")
    plt.ylabel("Average Length")
    plt.savefig(os.path.join("imgs", filename))

def run_centralized(num_of_episodes):
    env = CentralizedHunterEnv()
    agent = QLearning(env.action_space, gamma=0.95, alpha=0.1, epsilon=0.1, q_baseline=1)
    episode_lengths = np.zeros(num_of_episodes // 100 + 1)
    for episode in range(num_of_episodes):
        state = env.reset()
        done = False
        episode_length = 0
        while not done:
            action = agent.sample_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            episode_length += 1
        if (episode + 1) % 100 == 0 or episode == 0:
            episode_lengths[(episode + 1) // 100] = episode_length
    return episode_lengths

def run_decentralized_cooperative(num_of_episodes):
    env = HunterEnv()
    agent_1 = QLearning(env.single_agent_action_space, gamma=0.95, alpha=0.1, epsilon=0.1, q_baseline=1)
    agent_2 = QLearning(env.single_agent_action_space, gamma=0.95, alpha=0.1, epsilon=0.1, q_baseline=1)
    episode_lengths = np.zeros(num_of_episodes // 100 + 1)
    for episode in range(num_of_episodes):
        state = env.reset()
        done = False
        episode_length = 0
        while not done:
            action_1 = agent_1.sample_action(state)
            action_2 = agent_2.sample_action(state)
            action = (action_1, action_2)
            next_state, reward, done = env.step(action)
            agent_1.learn(state, action_1, reward[0], next_state, done)
            agent_2.learn(state, action_2, reward[1], next_state, done)
            state = next_state
            episode_length += 1
        if (episode + 1) % 100 == 0 or episode == 0:
            episode_lengths[(episode + 1) // 100] = episode_length
    return episode_lengths

def run_decentralized_competitive(num_of_episodes):
    env = HunterAndPreyEnv()
    agent_prey = QLearning(env.single_agent_action_space, gamma=0.95, alpha=0.1, epsilon=0.1, q_baseline=1)
    agent_hunter_1 = QLearning(env.single_agent_action_space, gamma=0.95, alpha=0.1, epsilon=0.1, q_baseline=1)
    agent_hunter_2 = QLearning(env.single_agent_action_space, gamma=0.95, alpha=0.1, epsilon=0.1, q_baseline=1)
    episode_lengths = np.zeros(num_of_episodes // 100 + 1)
    for episode in range(num_of_episodes):
        state = env.reset()
        done = False
        episode_length = 0
        while not done:
            action_hunter_1 = agent_hunter_1.sample_action(state)
            action_hunter_2 = agent_hunter_2.sample_action(state)
            action_prey = agent_prey.sample_action(state)
            action = (action_hunter_1, action_hunter_2, action_prey)
            next_state, reward, done = env.step(action)
            agent_hunter_1.learn(state, action_hunter_1, reward[0], next_state, done)
            agent_hunter_2.learn(state, action_hunter_2, reward[1], next_state, done)
            agent_prey.learn(state, action_prey, reward[2], next_state, done)
            state = next_state
            episode_length += 1
        if (episode + 1) % 100 == 0 or episode == 0:
            episode_lengths[(episode + 1) // 100] = episode_length
    return episode_lengths

if __name__ == '__main__':
    num_of_experiments = 30
    num_of_episodes = 50000

    avg_episode_lengths = np.zeros(num_of_episodes // 100 + 1)
    for i in range(num_of_experiments):
        print("Experiment: ", i + 1)
        # episode_lengths = run_centralized(num_of_episodes)
        episode_lengths = run_decentralized_cooperative(num_of_episodes)
        avg_episode_lengths += (episode_lengths - avg_episode_lengths) / (i + 1)
    
    # TODO: Save data in json file, because it takes a lot of time to run
    plot_average_length(avg_episode_lengths, "centralized_hunter_lengths.png")