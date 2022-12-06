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

        self.agent_position = np.zeros(2).astype(np.int16)

        self.x_choices = np.arange(int(0.7 * self.size[0]), self.size[0]).astype(np.int16)
        self.y_choices = np.arange(int(0.7 * self.size[1]), self.size[1]).astype(np.int16)
        self.pdf_x = np.linspace(0, 2, len(self.x_choices))
        self.pdf_x = sps.softmax(self.pdf_x)
        self.pdf_y = np.linspace(0, 2, len(self.y_choices))
        self.pdf_y = sps.softmax(self.pdf_y)
        self.goal_position = np.array([np.random.choice(self.x_choices, size=1, p=self.pdf_x), np.random.choice(self.y_choices, size=1, p=self.pdf_y)])

        self.action_space = Discrete(4)

        self.observation_space = Box(low=0,
                                     high=self.size,
                                     shape=(2,),
                                     dtype=np.int16)


    def step(self, action):
        reward = 0.

        if action == self.UP:
            self.agent_position[1] += 1
            if self.agent_position[1] > self.size[1]:
                reward -= 25.
                self.agent_position[1] = self.size[1]
        if action == self.DOWN:
            self.agent_position[1] -= 1
            if self.agent_position[1] < 0:
                reward -= 25.
                self.agent_position[1] = 0
        if action == self.LEFT:
            self.agent_position[0] -= 1
            if self.agent_position[0] < 0:
                reward -= 25.
                self.agent_position[0] = 0
        if action == self.RIGHT:
            self.agent_position[0] += 1
            if self.agent_position[0] > self.size[0]:
                reward -= 25.
                self.agent_position[0] = self.size[0]

        if self.agent_position[0] >= self.x_choices[0] and self.agent_position[1] >= self.y_choices[0]:
            reward += 2

        done = bool((self.agent_position == self.goal_position).all())
        reward = 100 if done else reward

        # self.goal_position = np.array([np.random.choice(self.x_choices, size=1, p=self.pdf_x), np.random.choice(self.y_choices, size=1, p=self.pdf_y)])
        return self.agent_position, reward, done, {}


    def save_episode(self, states, render_dir, update):
        xs = -0.5 + np.arange(0, self.size[0] + 1) * 1
        ys = -0.5 + np.arange(0, self.size[1] + 1) * 1

        w = 1
        h = 1

        _, ax = plt.subplots(1, 1)

        for x in xs:
            ax.plot([x, x], [ys[0], ys[-1]], color='black', alpha=.33, linestyle=':')
        for y in ys:
            ax.plot([xs[0], xs[-1]], [y, y], color='black', alpha=.33, linestyle=':')

        ax.plot(states[:, 0], states[:, 1], 'k-')
        ax.plot(self.goal_position[0], self.goal_position[1], 'b*', markersize=10)
        ax.set_xlim(0, self.size[0])
        ax.set_ylim(0, self.size[1])
        # plt.pause(0.01)
        plt.savefig(render_dir + '{}.png'.format(update), dpi=300)
        plt.close()


    def reset(self):
        self.agent_position = np.zeros(2).astype(np.int16)
        self.goal_position = np.array([np.random.choice(self.x_choices, size=1, p=self.pdf_x), np.random.choice(self.y_choices, size=1, p=self.pdf_y)])

        return self.agent_position

