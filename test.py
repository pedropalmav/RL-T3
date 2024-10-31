from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv
from agents.r_max import RMax
env = EscapeRoomEnv()
state = env.reset()
agent = RMax(env.action_space,R_max= 1, k = 10, gamma = 1.0, threshold=0.01, max_iterations=10)
for episode in range(1): #num episodes
    print(f'episode: {episode}')
    for i in range(200):#while not done:
        print(f'state: {state}')
        action = agent.sample_action(state) #run value iter 
        print(f'action: {action}')
        next_state, reward, done = env.step(action)
        print(f'next_state: {next_state}')
        agent.learn(state, action, reward, next_state)
        state = next_state
        