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
import datetime


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


    def select_action(self, state, eval=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            state /= self.env_size
            action, action_logprob, action_prob = self.policy_old.get_action(state)

        self.mem_buffer.states.append(state)
        self.mem_buffer.actions.append(action)
        self.mem_buffer.logprobs.append(action_logprob)

        if eval:
            return action.item(), action_prob.cpu().numpy()
        else:
            return action.item(), None


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


if __name__ == "__main__":
    env_name = 'gym-grid-v0'
    save_name = '50_static'
    evaluate = True

    run_name = f"{env_name}_{save_name}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    max_ep_len = 150        # 150 for size=50, 50 for size=16
    max_training_timesteps = 3e5

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    num_epochs = 40               # update policy for K epochs in one PPO update

    entropy_coeff = 0.05    # entropy coefficient to encourage exploration

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    dir = f'models/{save_name}/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    checkpoint_path = dir + 'PPO_model'

    render_dir = 'episode_renders/{}_{}/'.format(save_name, datetime.datetime.now().strftime("%m-%d_%H-%M-%S"))
    if not os.path.exists(render_dir):
        os.makedirs(render_dir)

    agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, num_epochs, eps_clip, entropy_coeff, max(env.observation_space.high))
    if evaluate:
        agent.load(checkpoint_path)
        max_training_timesteps = max_ep_len * 100

    start_time = time.time()

    time_step = 0
    i_episode = 0
    ACTION_PDF = []
    STATES = []
    while time_step <= max_training_timesteps:
        states = []
        state = env.reset()
        states.append(state.copy())
        ep_reward = 0

        for t in range(1, max_ep_len+1):
            action, action_pdf = agent.select_action(state, evaluate)
            if evaluate:
                ACTION_PDF.append(action_pdf.copy())
                STATES.append(state.copy())
            state, reward, done, _ = env.step(action)
            states.append(state.copy())

            agent.mem_buffer.rewards.append(reward)
            agent.mem_buffer.is_terminals.append(done)

            time_step += 1
            ep_reward += reward

            if time_step % update_timestep == 0 and not evaluate:
                agent.update()

            writer.add_scalar('performance/reward', reward, time_step)

            if time_step % 100000 == 0:
                print()
                print('-------------------')
                print('Saving model at: ', checkpoint_path)
                agent.save(checkpoint_path)

            if done:
                break

        states = np.asarray(states, dtype=np.int16)
        env.save_episode(states, render_dir, i_episode)

        print('Episode: {}, Reward: {}'.format(i_episode, ep_reward))
        i_episode += 1

    ACTION_PDF = np.asarray(ACTION_PDF)
    STATES = np.asarray(STATES)
    pickle.dump(STATES, open(checkpoint_path + '_states.pkl', 'wb'))
    pickle.dump(ACTION_PDF, open(checkpoint_path + '_action_pdf.pkl', 'wb'))

    env.close()
    writer.close()

    print('Total time for training: ', time.time() - start_time, ' seconds')
