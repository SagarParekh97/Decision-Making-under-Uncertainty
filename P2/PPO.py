import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()

parser.add_argument("--env-name", type=str, default="CartPole-v1",
                    help="the id of the gym environment")
parser.add_argument("--save_name", type=str, default=None,
                    help="name of the training run")
parser.add_argument("--total-timesteps", type=int, default=25000,
                    help="total timesteps of the experiments")
parser.add_argument("--num_envs", type=int, default=5,
                    help="number of environments to use in vectorized env")
parser.add_argument("--num-steps", type=int, default=128,
                    help="the number of steps to run in each environment per policy rollout")
parser.add_argument('--update_epochs', type=int, default=1,
                    help='number of times to update each iteration')

parser.add_argument('--lr', type=float, default=2.5e-4,
                    help='learning rate of the optimizer')
parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                    help="Use GAE for advantage computation")
parser.add_argument("--gamma", type=float, default=0.99,
                    help="the discount factor gamma")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="the lambda for the general advantage estimation")
parser.add_argument("--num-minibatches", type=int, default=4,
                    help="the number of mini-batches")
parser.add_argument("--clip-coef", type=float, default=0.2,
                    help="the surrogate clipping coefficient")
parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                    help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                    help="Toggles advantages normalization")
parser.add_argument("--ent-coef", type=float, default=0.01,
                    help="coefficient of the entropy")
parser.add_argument("--vf-coef", type=float, default=0.5,
                    help="coefficient of the value function")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="the maximum norm for the gradient clipping")

args = parser.parse_args()
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def layer_init(layer, std=np.sqrt(2), bias_cont=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_cont)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    run_name = f"{args.env_name}_{args.save_name}_{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")

    envs = gym.vector.make(args.env_name, num_envs=args.num_envs)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    total_steps = 0
    start_time = time.time()
    next_state = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        for step in range(0, args.num_steps):
            total_steps += 1 * args.num_envs
            obs[step] = next_state
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, = agent.get_action_and_value(next_state)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_state, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_state, next_done = torch.Tensor(next_state).to(device), torch.Tensor(done).to(device)

        with torch.no_grad():
            next_value = agent.get_value(next_state).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)


        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                        )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], total_steps)
        writer.add_scalar("losses/value_loss", v_loss.item(), total_steps)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), total_steps)
        writer.add_scalar("losses/entropy", entropy_loss.item(), total_steps)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), total_steps)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), total_steps)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), total_steps)
        writer.add_scalar("losses/explained_variance", explained_var, total_steps)
        print("SPS:", int(total_steps / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(total_steps / (time.time() - start_time)), total_steps)
        writer.add_scalar("performance/rewards", round(reward, ))
    envs.close()
    writer.close()