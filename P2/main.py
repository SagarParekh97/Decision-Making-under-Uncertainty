import gym
import torch
import numpy as np
from PPO_ref import device, PPO
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from datetime import datetime


EnvName = 'CartPole-v1'
# EnvName = 'GridWorld-v0'
env_with_Dead = [True, True]
env = gym.make(EnvName)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
max_e_steps = env._max_episode_steps

timenow = str(datetime.now())[0:-10]
timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
writepath = 'runs/{}'.format(EnvName) + timenow
if os.path.exists(writepath): shutil.rmtree(writepath)
writer = SummaryWriter(log_dir=writepath)

print('Env:', EnvName, '  state_dim:', state_dim, '  action_dim:', action_dim, '  max_e_steps:', max_e_steps)
print('\n')

kwargs = {
    "env_with_Dead": EnvName,
    "state_dim": state_dim,
    "action_dim": action_dim,
    "gamma": 0.99,
    "lambd": 0.95,
    "hidden_size": 64,
    "lr": 1e-4,
    "clip_rate": 0.2,
    "K_epochs": 10,
    "batch_size": 64,
    "l2_reg": 0,
    "entropy_coef": 0.2,  #hard env needs large value
    "adv_normalization": False,
    "entropy_coef_decay": 0.99,
}

if not os.path.exists('model'): os.mkdir('model')
model = PPO(**kwargs)
# if Loadmodel: model.load(ModelIdex)



'''

'''
render = False
T_horizon = 2048
'''

'''


traj_length = 0
total_steps = 0
while total_steps < 5e6:
    s, done, steps, ep_r = env.reset()[0], False, 0, 0

    '''Interact & train'''
    while not done:
        traj_length += 1
        steps += 1
        a, pi_a = model.select_action(torch.from_numpy(s).float().to(device))

        s_prime, r, done, info, _ = env.step(a)

        if (done and steps != max_e_steps):
            dw = True  # dw: dead and win
        else:
            dw = False

        model.put_data((s, a, r, s_prime, pi_a, done, dw))
        s = s_prime
        ep_r += r

        '''update if its time'''
        if not render:
            if traj_length % T_horizon == 0:
                a_loss, c_loss, entropy = model.update()
                traj_length = 0
                writer.add_scalar('a_loss', a_loss, global_step=total_steps)
                writer.add_scalar('c_loss', c_loss, global_step=total_steps)
                writer.add_scalar('entropy', entropy, global_step=total_steps)

        '''save model'''
        if total_steps % 5000 == 0:
            model.save(total_steps)

env.close()