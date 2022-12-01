import gym
from gym.spaces import Box, Discrete
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class gridWorld(gym.Env):
    # Actions available
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3

    def __init__(self, size=128):
        if type(size) is tuple:
            self.size = np.array(size)
        else:
            self.size = np.ones(2) * size

        self.agent_position = np.array([0, 0]).astype(np.uint8)

        self.goal_position = np.random.randint((self.size * 0.8).astype(int), self.size - np.ones(2))

        self.action_space = Discrete(4)

        self.observation_space = Box(low=0,
                                     high=self.size,
                                     shape=(2,),
                                     dtype=np.uint8)

    def step(self, action):
        reward = -0.1

        if action == self.UP:
            self.agent_position[1] += 1
        if action == self.DOWN:
            self.agent_position[1] -= 1
        if action == self.LEFT:
            self.agent_position[0] -= 1
        if action == self.RIGHT:
            self.agent_position[0] += 1

        self.agent_position = np.clip(self.agent_position, np.zeros(2), self.size).astype(np.uint8)

        done = bool((self.agent_position == self.goal_position).all())
        reward = 100 if done else reward

        return self.agent_position, reward, done, {}


    def render(self, mode="console"):
        xs = np.linspace(-0.5, self.size[0] + 0.5, self.size[0]+1)
        ys = np.linspace(-0.5, self.size[1] + 0.5, self.size[1]+1)
        w = 1
        h = 1

        ax = plt.gca()
        for i, x in enumerate(xs[:-1]):
            for j, y in enumerate(ys[:-1]):
                if i % 2 == j % 2:
                    ax.add_patch(Rectangle((x, y), w, h, fill=True, color='#838B8B', alpha=0.1))

        for x in xs:
            plt.plot([x, x], [ys[0], ys[-1]], color='black', alpha=.33, linestyle=':')
        for y in ys:
            plt.plot([xs[0], xs[-1]], [y, y], color='black', alpha=.33, linestyle=':')
        plt.show()


    def reset(self):
        self.agent_position = np.zeros(2).astype(np.uint8)

        return self.agent_position

