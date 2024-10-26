from Environments.SimpleEnvs.CliffEnv import CliffEnv
from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv

from agents.q_learning import QLearning
from agents.sarsa import Sarsa
from agents.n_step import NStep
from agents.dyna import DynaQ

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

def run_q_learning(env, num_episodes):
    agent = QLearning(env.action_space)
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.sample_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
        print(f"Episode {episode} finished")

def run_sarsa(env, num_episodes):
    agent = Sarsa(env.action_space)
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        action = agent.sample_action(state)
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.sample_action(next_state)
            agent.learn(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
        print(f"Episode {episode} finished")

def run_dyna(env, num_episodes):
    agent = DynaQ(env.action_space)
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.sample_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
        print(f"Episode {episode} finished")

# TODO: Refactor del año
def run_n_step(env, num_episodes):
    agent = NStep(env.action_space, 0.1, 0.9, 0.1, n=4)
    for episode in range(num_episodes):
        state = env.reset()
        agent.store_state(state)
        action = agent.sample_action(state)
        episode_len = float('inf')
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
            tau = t - agent.n + 1
            if tau >= 0:
            # TODO: Implement this in learn method
                G = sum(agent.reward_store[i] * (agent.gamma ** (i - tau - 1)) for i in range(tau + 1, min(tau + agent.n, episode_len)))
                if tau + agent.n < episode_len:
                    G += agent.get_q_value(agent.state_store[tau + agent.n], agent.action_store[tau + agent.n]) * (agent.gamma ** agent.n)
                agent.q_values[(agent.state_store[tau], agent.action_store[tau])] = agent.get_q_value(agent.state_store[tau], agent.action_store[tau]) + agent.alpha * (G - agent.get_q_value(agent.state_store[tau], agent.action_store[tau]))
            if tau == episode_len - 1:
                break
            t += 1
        print(f"Episode {episode} finished")


if __name__ == "__main__":
    env = CliffEnv()
    # env = EscapeRoomEnv()
    # run_q_learning(env, 500)
    # run_sarsa(env, 500)
    run_n_step(env, 500)
    # play_simple_env(env)

