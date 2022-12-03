import gym
from gym.spaces import Box, Discrete
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.special as sps


class gridWorld(gym.Env):
    # Actions available
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3

    def __init__(self, size=50):
        if type(size) is tuple:
            self.size = np.array(size)
        else:
            self.size = np.ones(2) * size

        self.agent_position = np.array([int(self.size[0] / 2), int(self.size[1] / 2)]).astype(np.uint8)

        self.x_choices = np.arange(int(0.7 * self.size[0]), self.size[0]).astype(np.uint8)
        self.y_choices = np.arange(int(0.7 * self.size[1]), self.size[1]).astype(np.uint8)
        self.pdf_x = np.linspace(0, 2, len(self.x_choices))
        self.pdf_x = sps.softmax(self.pdf_x)
        self.pdf_y = np.linspace(0, 2, len(self.x_choices))
        self.pdf_y = sps.softmax(self.pdf_y)
        self.goal_position = np.array([np.random.choice(self.x_choices, size=1, p=self.pdf_x), np.random.choice(self.y_choices, size=1, p=self.pdf_y)])

        self.action_space = Discrete(4)

        self.observation_space = Box(low=0,
                                     high=self.size,
                                     shape=(2,),
                                     dtype=np.uint8)

    def step(self, action):
        reward = 0.

        if action == self.UP:
            self.agent_position[1] += 1
        if action == self.DOWN:
            if self.agent_position[1] != 0:
                self.agent_position[1] -= 1
        if action == self.LEFT:
            if self.agent_position[0] != 0:
                self.agent_position[0] -= 1
        if action == self.RIGHT:
            self.agent_position[0] += 1

        self.agent_position = np.clip(self.agent_position, np.zeros(2), self.size).astype(np.uint8)

        done = bool((self.agent_position == self.goal_position).all())
        reward = 100 if done else reward

        # self.goal_position = np.array([np.random.choice(self.x_choices, size=1, p=self.pdf_x), np.random.choice(self.y_choices, size=1, p=self.pdf_y)])
        return self.agent_position, reward, done, {}


    def render(self, mode="console"):
        xs = -0.5 + np.arange(0, self.size[0] + 1) * 1
        ys = -0.5 + np.arange(0, self.size[1] + 1) * 1

        w = 1
        h = 1

        _, ax = plt.subplots(1, 1)
        # for i, x in enumerate(xs[:-1]):
        #     for j, y in enumerate(ys[:-1]):
        #         if i % 2 == j % 2:
        #             ax.add_patch(Rectangle((x, y), w, h, fill=True, color='#838B8B', alpha=0.1))

        for x in xs:
            ax.plot([x, x], [ys[0], ys[-1]], color='black', alpha=.33, linestyle=':')
        for y in ys:
            ax.plot([xs[0], xs[-1]], [y, y], color='black', alpha=.33, linestyle=':')

        ax.plot(self.agent_position[0], self.agent_position[1], 'rs', markersize=10)
        ax.plot(self.goal_position[0], self.goal_position[1], 'b*', markersize=10)
        plt.pause(0.01)
        return ax


    def reset(self):
        self.agent_position = np.array([int(self.size[0] / 2), int(self.size[1] / 2)]).astype(np.uint8)
        self.goal_position = np.array([np.random.choice(self.x_choices, size=1, p=self.pdf_x), np.random.choice(self.y_choices, size=1, p=self.pdf_y)])

        return self.agent_position

