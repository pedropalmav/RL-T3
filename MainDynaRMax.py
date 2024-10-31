
from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv

from agents.dyna import DynaQ
from agents.r_max import RMax
import numpy as np

def show(env, current_state, reward=None):
    env.show()
    print(f"Raw state: {current_state}")
    if reward:
        print(f"Reward: {reward}")


def run_dyna(env, num_episodes, gamma=1.0, alpha=0.5, epsilon=0.1, n=0):
    agent = DynaQ(env.action_space, alpha = alpha, gamma = gamma, epsilon = epsilon, n = n) 
    steps_per_episode = []
    reward_per_episode =[]
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        steps=0
        total_reward = 0
        while not done:
            action = agent.sample_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            steps+=1
            total_reward += reward
        reward_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        print(f"Episode {episode+1} finished")
    return reward_per_episode, steps_per_episode
        
def run_rmax(env, num_episodes):
    agent = RMax(env.action_space,R_max= -1, k = 3, gamma = 1.0, threshold=0.001, max_iterations=10)
    steps_per_episode = []
    reward_per_episode =[]
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        steps=0
        total_reward = 0
        while not done:
            action = agent.sample_action(state) 
            next_state, reward, done = env.step(action)
            total_reward += reward
            agent.learn(state, action, reward, next_state)
            state = next_state
            steps+=1
        reward_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        print(f"Episode {episode+1} finished in {steps} steps with reward {total_reward}")
    return reward_per_episode, steps_per_episode

if __name__ == "__main__":
    env = EscapeRoomEnv()
    dyna = False
    if dyna:
        n=1000
        total_reward = np.zeros(20)
        total_steps = np.zeros(20)
        for run in range(5):
            #start = time.time()
            print(f"Run {run+1}")
            reward_per_episode, steps_per_episode = run_dyna(env, 20, n=n)
            #reward_per_episode, steps_per_episode = run_rmax(env, 20)
            rwd = np.array(reward_per_episode)
            stp = np.array(steps_per_episode)
            total_reward += rwd
            total_steps += stp   
            #end = time.time()
        avg_reward = total_reward/5
        avg_steps = total_steps/5
        np.save(f'dyna_{n}_escape_room_reward.npy', avg_reward)
        np.save(f'dyna_{n}_escape_room_steps.npy', avg_steps) 
    else:
        reward_per_episode, steps_per_episode = run_rmax(env, 20)
        rwd = np.array(reward_per_episode)
        stp = np.array(steps_per_episode)
        np.save(f'rmax_escape_room_reward.npy', rwd)
        np.save(f'rmax_escape_room_steps.npy', stp)


