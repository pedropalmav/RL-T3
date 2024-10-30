import numpy as np

from Environments.PartiallyObservableEnvs.InvisibleDoorEnv import InvisibleDoorEnv
from MainSimpleEnvs import show, get_action_from_user, play_simple_env
from MemoryWrappers.BinaryMemory import BinaryMemory
from MemoryWrappers.KOrderMemory import KOrderMemory
from MemoryWrappers.KOrderBuffer import KOrderBuffer

from agents.q_learning import QLearning
from agents.sarsa import Sarsa
from agents.n_step import NStep

from MainMultiGoal import plot_average_length


def play_env_with_binary_memory():
    num_of_bits = 1
    env = InvisibleDoorEnv()
    env = BinaryMemory(env, num_of_bits)

    key2action = {'a': "left", 'd': "right", 'w': "up", 's': "down"}
    key2memory = {str(i): i for i in range(2**num_of_bits)}
    s = env.reset()
    show(env, s)
    done = False
    while not done:
        print("Environment action: ", end="")
        env_action = get_action_from_user(key2action)
        print(f"Memory action ({', '.join(key2memory.keys())}): ", end="")
        mem_action = get_action_from_user(key2memory)
        action = env_action, mem_action
        s, r, done = env.step(action)
        show(env, s, r)


def play_env_with_k_order_memory():
    memory_size = 2
    env = InvisibleDoorEnv()
    env = KOrderMemory(env, memory_size)
    play_simple_env(env)


def play_env_without_extra_memory():
    env = InvisibleDoorEnv()
    play_simple_env(env)

def run_q_learning(env, num_of_episodes):
    agent = QLearning(env.action_space, gamma=0.99, alpha=0.1, epsilon=0.01, q_baseline=1)
    episode_lengths = np.zeros(num_of_episodes)
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
        episode_lengths[episode] = episode_length
    return episode_lengths

def run_sarsa(env, num_episodes):
    agent = Sarsa(env.action_space, gamma=0.99, alpha=0.1, epsilon=0.01, q_baseline=1)
    episode_lengths = np.zeros(num_episodes)
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        action = agent.sample_action(state)
        episode_length = 1
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.sample_action(next_state)
            agent.learn(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
            episode_length += 1
        episode_lengths[episode] = episode_length
    return episode_lengths

def run_n_step(env, num_episodes):
    agent = NStep(env.action_space, 16, gamma=0.99, alpha=0.1, epsilon=0.01, q_baseline=1)
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

def select_memory_wrapper():
    print("Select memory wrapper:")
    print("1. No memory")
    print("2. K-order memory")
    print("3. Binary memory")
    print("4. K-order buffer")

    memory_wrapper = int(input())
    if memory_wrapper == 1:
        return InvisibleDoorEnv(), "no_memory"
    elif memory_wrapper == 2:
        memory_size = 2
        return KOrderMemory(InvisibleDoorEnv(), memory_size), f"{memory_size}_order_memory"
    elif memory_wrapper == 3:
        num_of_bits = 1
        return BinaryMemory(InvisibleDoorEnv(), num_of_bits), "binary_memory"
    elif memory_wrapper == 4:
        buffer_size = 1
        return KOrderBuffer(InvisibleDoorEnv(), buffer_size), f"{buffer_size}_order_buffer"
    else:
        print("Invalid input")

if __name__ == '__main__':
    env, filename_prefix = select_memory_wrapper()
    num_of_experiments = 30
    num_of_episodes = 1000
    baselines = [run_q_learning, run_sarsa, run_n_step]
    baselines_avg_lengths = {}
    for baseline in baselines:
        average_lengths = np.zeros(num_of_episodes)
        for i in range(num_of_experiments):
            lengths = baseline(env, num_of_episodes)
            average_lengths += (lengths - average_lengths) / (i + 1)
            print(f"Experiment {i + 1} finished")
        baselines_avg_lengths[baseline.__name__[4:]] = average_lengths
    plot_average_length(baselines_avg_lengths, f"{filename_prefix}_eps_lengths.png")