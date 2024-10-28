from collections import deque

from Environments.AbstractEnv import AbstractEnv

class KOrderBuffer(AbstractEnv):

    def __init__(self, env, buffer_size):
        self.__env = env
        self.__buffer_size = buffer_size
        self.__current_state = None
        self.__memory = None

    @property
    def action_space(self):
        env_actions = self.__env.action_space
        memory_actions = ["save", "ignore"]
        return [(env_action, memory_action) for env_action in env_actions for memory_action in memory_actions]

    def reset(self):
        s = self.__env.reset()
        self.__memory = deque(maxlen=self.__buffer_size)
        self.__current_state = s
        return tuple(self.__memory)

    def step(self, action):
        env_action, mem_action = action
        if mem_action == "save":
            self.__memory.append(self.__current_state)
        s, r, done = self.__env.step(env_action)
        self.__current_state = s
        return (s, tuple(self.__memory)), r, done

    def show(self):
        self.__env.show()
        print(f"Memory: {self.__memory}")