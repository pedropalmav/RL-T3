import numpy as np
import matplotlib.pyplot as plt

from Environments.PartiallyObservableEnvs.InvisibleDoorEnv import InvisibleDoorEnv
from MainSimpleEnvs import show, get_action_from_user, play_simple_env
from MemoryWrappers.BinaryMemory import BinaryMemory
from MemoryWrappers.KOrderMemory import KOrderMemory
from MemoryWrappers.KOrderBuffer import KOrderBuffer

from MainMultiGoal import run_q_learning, run_sarsa, run_n_step, plot_average_length


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