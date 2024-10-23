from Environments.SimpleEnvs.CliffEnv import CliffEnv
from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv

from agents.q_learning import QLearning


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

def run_agent(agent, env, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.sample_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
        print(f"Episode {episode} finished")


if __name__ == "__main__":
    env = CliffEnv()
    # env = EscapeRoomEnv()
    q_learning_agent = QLearning(env.action_space)
    run_agent(q_learning_agent, env, 500)
    # play_simple_env(env)

