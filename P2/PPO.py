import os
import pickle
import time
import gym
import gym_grid
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from sympy.utilities.iterables import multiset_permutations
import sys


# select the device to train on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class to save the agent's experience buffer
class Memory_buffer():
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []


# create the NN architecture for the PPO agent
class Agent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Agent, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def get_action(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach().cpu(), action_logprob.detach().cpu(), action_probs.detach().cpu()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        value = self.critic(state)

        return action_logprobs, value, dist_entropy


# Class that trains the policy
class PPO(object):
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, num_epochs, eps_clip, entropy, env_size):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.num_epochs = num_epochs
        self.entropy = entropy
        self.env_size = env_size

        self.mem_buffer = Memory_buffer()

        self.policy = Agent(state_dim, action_dim).to(device)
        self.optim = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = Agent(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss = nn.MSELoss()


    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            state /= self.env_size
            action, action_logprob, action_prob = self.policy_old.get_action(state)

        self.mem_buffer.states.append(state)
        self.mem_buffer.actions.append(action)
        self.mem_buffer.logprobs.append(action_logprob)

        return action.item(), action_prob.cpu().numpy()


    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(self.mem_buffer.rewards), reversed(self.mem_buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.mem_buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.mem_buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.mem_buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.num_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.loss(state_values, rewards) - self.entropy*dist_entropy

            # take gradient step
            self.optim.zero_grad()
            loss.mean().backward()
            self.optim.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.mem_buffer.clear()


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)


    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def balance(b1, b2):
    ret = 1 - abs(b1 - b2) / (b1 + b2 + 1e-7)
    return ret



if __name__ == "__main__":
    env_name = 'gym-grid-v0'        # name of the environment you want to use
    save_name = 'RL'        # save the file
    uncertainty_aware = sys.argv[1]           # use entropy-aware design
    exp_exp = sys.argv[2]
    if exp_exp == 'true':
        save_name += 'G'
    elif exp_exp == 'false':
        save_name += 'R'

    if uncertainty_aware == 'true':
        uncertainty_aware = True
        save_name += '_{}'.format(sys.argv[3])

    save_name += '40'

    # log data
    run_name = f"{env_name}_{save_name}"
    writer = SummaryWriter(f"runs/{run_name}")

    max_ep_len = 400    # time horizon
    max_training_timesteps = max_ep_len * 2000      # training episodes

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    num_epochs = 40               # update policy for K epochs in one PPO update

    entropy_coeff = 0.05    # entropy coefficient to encourage exploration
    epsilon_0 = 0.05

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    env = gym.make(env_name)
    W = env.action_space.n      # parameter W for uncertainty
    a = 1 / 4                   # base rate, assumed uniform

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    state_dim = env.observation_space.shape[0]  # dimension of observation space
    action_dim = env.action_space.n             # dimension of action space

    # save trained models
    dir = f'models/{save_name}/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    checkpoint_path = dir + 'PPO_model'

    # save each episode render
    render_dir = 'episode_renders/{}/'.format(save_name)
    if not os.path.exists(render_dir):
        os.makedirs(render_dir)

    # PPO agent
    agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, num_epochs, eps_clip, entropy_coeff, max(env.observation_space.high))

    start_time = time.time()

    time_step = 0
    i_episode = 0
    ACTION_PDF = []
    STATES = []
    ENTROPY = []
    VACUITY = []
    DISSONANCE = []
    REWARDS = []
    REACHED_GOAL = []
    entropy = 0.
    dissonance = 0.
    u_dd = 0.
    # the agent interacts with the environment for the given number of episodes
    while time_step <= max_training_timesteps:
        states = []
        state = env.reset()
        states.append(state.copy())
        ep_reward = 0
        ep_terminal = 0

        for t in range(1, max_ep_len+1):
            action, action_pdf = agent.select_action(state)
            ACTION_PDF.append(action_pdf.copy())
            STATES.append(state.copy())

            state, reward, done, _ = env.step(action)
            states.append(state.copy())
            if uncertainty_aware:
                if sys.argv[2] == 'true':
                    if sys.argv[3] == 'H':
                        entropy_coeff = epsilon_0 * (1 - entropy)
                    elif sys.argv[3] == 'D':
                        entropy_coeff = epsilon_0 * (1 - dissonance)
                    elif sys.argv[3] == 'V':
                        entropy_coeff = epsilon_0 * (1 - u_dd)
                else:
                    if sys.argv[3] == 'H':
                        reward += 1 - entropy
                    elif sys.argv[3] == 'D':
                        reward += 1 - dissonance
                    elif sys.argv[3] == 'V':
                        reward += 1 - u_dd

            agent.mem_buffer.rewards.append(reward)
            agent.mem_buffer.is_terminals.append(done)

            time_step += 1
            ep_reward += reward

            if time_step % update_timestep == 0:
                agent.update()

            if time_step % 100000 == 0:
                print()
                print('-------------------')
                print('Saving model at: ', checkpoint_path)
                agent.save(checkpoint_path)

            if done:
                # check if the agetn reached the goal
                if (env.agent_position - env.goal_position).all():
                    ep_terminal = 1
                else:
                    ep_terminal = 0
                break

        REACHED_GOAL.append(ep_terminal)

        # calculate the uncertainty using saved experience
        dirichlet = np.asarray(ACTION_PDF)
        evidence = np.argmax(dirichlet, axis=1)
        unique, counts = np.unique(evidence, return_counts=True)
        r = dict(zip(unique, counts))
        belief = {0: 0., 1: 0., 2: 0., 3: 0.}
        proj_prob = {0: 0., 1: 0., 2: 0., 3: 0.}
        u = W / (W + len(evidence))
        VACUITY.append(u)
        entropy = 0.
        # calculate the belief, projected probability, and entropy
        for key in r:
            belief[key] = r[key] / (W + len(evidence))
            proj_prob[key] = belief[key] + a * u
            entropy -= proj_prob[key] * (np.log(proj_prob[key]) / np.log(4))
        ENTROPY.append(entropy)

        u_dd = np.inf
        for k in proj_prob:
            u_dd = min(u_dd, proj_prob[k]/a)

        # calculate dissonance
        bal = {}
        permutations = multiset_permutations(range(4), 2)
        for p in permutations:
            bal[str(p)] = balance(belief[p[0]], belief[p[1]])

        dissonance = 0.
        set_without_xi = [0, 1, 2, 3]
        for xi in range(4):
            set_without_xi.remove(xi)
            num = 0.
            den = 1e-7
            for xj in set_without_xi:
                num += belief[xj] * bal[f'[{int(xi)}, {int(xj)}]']
                den += belief[xj]
            dissonance += belief[xi] * num / den

            set_without_xi = [0, 1, 2, 3]
        DISSONANCE.append(dissonance)
        REWARDS.append(ep_reward)

        # render the episode
        states = np.asarray(states, dtype=np.int16)
        if i_episode > max_training_timesteps - 100:
            env.save_episode(states, render_dir, i_episode)

        writer.add_scalar('performance/reward', ep_reward, i_episode)

        print('Episode: {}, Reward: {}'.format(i_episode, ep_reward))
        i_episode += 1

    ACTION_PDF = np.asarray(ACTION_PDF)
    STATES = np.asarray(STATES)
    ENTROPY = np.asarray(ENTROPY)
    VACUITY = np.asarray(VACUITY)
    DISSONANCE = np.asarray(DISSONANCE)
    REWARDS = np.asarray(REWARDS)
    REACHED_GOAL = np.asarray(REACHED_GOAL)
    pickle.dump(STATES, open(checkpoint_path + '_states.pkl', 'wb'))
    pickle.dump(ACTION_PDF, open(checkpoint_path + '_action_pdf.pkl', 'wb'))
    pickle.dump(DISSONANCE, open(checkpoint_path + '_dissonance.pkl', 'wb'))
    pickle.dump(ENTROPY, open(checkpoint_path + '_entropy.pkl', 'wb'))
    pickle.dump(VACUITY, open(checkpoint_path + '_vacuity.pkl', 'wb'))
    pickle.dump(REWARDS, open(checkpoint_path + '_rewards.pkl', 'wb'))
    pickle.dump(REACHED_GOAL, open(checkpoint_path + '_reached_goal.pkl', 'wb'))

    env.close()
    writer.close()

    print('Total time for training: ', time.time() - start_time, ' seconds')
