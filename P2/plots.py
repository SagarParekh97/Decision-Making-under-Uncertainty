# import os
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
#
#
#
# def EMA_filter(time_series, weight=0.99):
#     last = time_series[0]
#     smoothed = []
#     for p in time_series:
#         filt_val = weight * last + (1 - weight) * p
#         smoothed.append(filt_val)
#         last = filt_val
#
#     return np.array(smoothed)
#
#
# dir1 = 'models/results/'
# dir2 = 'models/results_entropy/'
# dir3 = 'models/results_dissonance/'
# dir4 = 'models/results_entropy_max/'
#
# diss1 = pickle.load(open(dir1 + 'PPO_model_dissonance.pkl', 'rb'))
# diss2 = pickle.load(open(dir2 + 'PPO_model_dissonance.pkl', 'rb'))
# diss3 = pickle.load(open(dir3 + 'PPO_model_dissonance.pkl', 'rb'))
# diss4 = pickle.load(open(dir4 + 'PPO_model_dissonance.pkl', 'rb'))
#
# entropy1 = pickle.load(open(dir1 + 'PPO_model_entropy.pkl', 'rb'))
# entropy2 = pickle.load(open(dir2 + 'PPO_model_entropy.pkl', 'rb'))
# entropy3 = pickle.load(open(dir3 + 'PPO_model_entropy.pkl', 'rb'))
# entropy4 = pickle.load(open(dir4 + 'PPO_model_entropy.pkl', 'rb'))
#
# vacuity1 = pickle.load(open(dir1 + 'PPO_model_vacuity.pkl', 'rb'))
# vacuity2 = pickle.load(open(dir2 + 'PPO_model_vacuity.pkl', 'rb'))
# vacuity3 = pickle.load(open(dir3 + 'PPO_model_vacuity.pkl', 'rb'))
# vacuity4 = pickle.load(open(dir4 + 'PPO_model_vacuity.pkl', 'rb'))
#
# reward1 = pickle.load(open(dir1 + 'PPO_model_rewards.pkl', 'rb'))
# reward2 = pickle.load(open(dir2 + 'PPO_model_rewards.pkl', 'rb'))
# reward3 = pickle.load(open(dir3 + 'PPO_model_rewards.pkl', 'rb'))
# reward4 = pickle.load(open(dir4 + 'PPO_model_rewards.pkl', 'rb'))
#
# reached_goal1 = pickle.load(open(dir1 + 'PPO_model_reached_goal.pkl', 'rb'))
# reached_goal2 = pickle.load(open(dir2 + 'PPO_model_reached_goal.pkl', 'rb'))
# reached_goal3 = pickle.load(open(dir3 + 'PPO_model_reached_goal.pkl', 'rb'))
# reached_goal4 = pickle.load(open(dir4 + 'PPO_model_reached_goal.pkl', 'rb'))
#
# if not os.path.exists('plots/'):
#     os.mkdir('plots/')
#
# episodes1 = range(1, len(diss1)+1)
# episodes2 = range(1, len(diss2)+1)
# episodes3 = range(1, len(diss3)+1)
# episodes4 = range(1, len(diss4)+1)
# plt.plot(episodes1, diss1, label='RL')
# plt.plot(episodes2, diss2, label='RL-H')
# plt.plot(episodes3, diss3, label='RL-d')
# plt.plot(episodes4, diss4, label='RL-Hmax')
# plt.legend()
# plt.xlabel('Episodes')
# plt.ylabel('Dissonance')
# plt.savefig('plots/Dissonance.png', dpi=300)
# plt.close()
#
# plt.plot(episodes1, entropy1, label='RL')
# plt.plot(episodes2, entropy2, label='RL-H')
# plt.plot(episodes3, entropy3, label='RL-d')
# plt.plot(episodes4, entropy4, label='RL-Hmax')
# plt.legend()
# plt.xlabel('Episodes')
# plt.ylabel('Entropy')
# plt.savefig('plots/Entropy.png', dpi=300)
# plt.close()
#
# plt.plot(episodes1, vacuity1, label='RL')
# plt.plot(episodes2, vacuity2, label='RL-H')
# plt.plot(episodes3, vacuity3, label='RL-d')
# plt.plot(episodes4, vacuity4, label='RL-Hmax')
# plt.legend()
# plt.xlabel('Episodes')
# plt.ylabel('Vacuity')
# plt.savefig('plots/Vacuity.png', dpi=300)
# plt.close()
#
# plt.plot(episodes1, EMA_filter(reward1), label='RL')
# plt.plot(episodes2, EMA_filter(reward2), label='RL-H')
# plt.plot(episodes3, EMA_filter(reward3), label='RL-d')
# plt.plot(episodes4, EMA_filter(reward4), label='RL-Hmax')
# plt.legend()
# plt.xlabel('Episodes')
# plt.ylabel('Rewards')
# plt.savefig('plots/Rewards.png', dpi=300)
# plt.close()
#
# plt.bar(['RL', 'RL-H', 'RL-d', 'RL-Hmax'], [sum(reached_goal1), sum(reached_goal2), sum(reached_goal3), sum(reached_goal4)])
# plt.savefig('plots/efficiency.png', dpi=300)
# plt.close()


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt



def EMA_filter(time_series, weight=0.99):
    last = time_series[0]
    smoothed = []
    for p in time_series:
        filt_val = weight * last + (1 - weight) * p
        smoothed.append(filt_val)
        last = filt_val

    return np.array(smoothed)


dir1 = 'models/results_08_entropy_max/'
dir2 = 'models/results_09_entropy_max/'
dir3 = 'models/results_entropy_max/'

diss1 = pickle.load(open(dir1 + 'PPO_model_dissonance.pkl', 'rb'))
diss2 = pickle.load(open(dir2 + 'PPO_model_dissonance.pkl', 'rb'))
diss3 = pickle.load(open(dir3 + 'PPO_model_dissonance.pkl', 'rb'))

entropy1 = pickle.load(open(dir1 + 'PPO_model_entropy.pkl', 'rb'))
entropy2 = pickle.load(open(dir2 + 'PPO_model_entropy.pkl', 'rb'))
entropy3 = pickle.load(open(dir3 + 'PPO_model_entropy.pkl', 'rb'))

vacuity1 = pickle.load(open(dir1 + 'PPO_model_vacuity.pkl', 'rb'))
vacuity2 = pickle.load(open(dir2 + 'PPO_model_vacuity.pkl', 'rb'))
vacuity3 = pickle.load(open(dir3 + 'PPO_model_vacuity.pkl', 'rb'))

reward1 = pickle.load(open(dir1 + 'PPO_model_rewards.pkl', 'rb'))
reward2 = pickle.load(open(dir2 + 'PPO_model_rewards.pkl', 'rb'))
reward3 = pickle.load(open(dir3 + 'PPO_model_rewards.pkl', 'rb'))

reached_goal1 = pickle.load(open(dir1 + 'PPO_model_reached_goal.pkl', 'rb'))
reached_goal2 = pickle.load(open(dir2 + 'PPO_model_reached_goal.pkl', 'rb'))
reached_goal3 = pickle.load(open(dir3 + 'PPO_model_reached_goal.pkl', 'rb'))

if not os.path.exists('plots/'):
    os.mkdir('plots/')

episodes1 = range(1, len(diss1)+1)
episodes2 = range(1, len(diss2)+1)
episodes3 = range(1, len(diss3)+1)
plt.plot(episodes1, diss1, label='gamma = 0.8')
plt.plot(episodes2, diss2, label='gamma = 0.9')
plt.plot(episodes3, diss3, label='gamma = 0.99')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Dissonance')
plt.savefig('plots/Dissonance.png', dpi=300)
plt.close()

plt.plot(episodes1, entropy1, label='gamma = 0.8')
plt.plot(episodes2, entropy2, label='gamma = 0.9')
plt.plot(episodes3, entropy3, label='gamma = 0.99')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Entropy')
plt.savefig('plots/Entropy.png', dpi=300)
plt.close()

plt.plot(episodes1, vacuity1, label='gamma = 0.8')
plt.plot(episodes2, vacuity2, label='gamma = 0.9')
plt.plot(episodes3, vacuity3, label='gamma = 0.99')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Vacuity')
plt.savefig('plots/Vacuity.png', dpi=300)
plt.close()

plt.plot(episodes1, EMA_filter(reward1), label='gamma = 0.8')
plt.plot(episodes2, EMA_filter(reward2), label='gamma = 0.9')
plt.plot(episodes3, EMA_filter(reward3), label='gamma = 0.99')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig('plots/Rewards.png', dpi=300)
plt.close()

plt.bar(['gamma = 0.8', 'gamma = 0.9', 'gamma = 0.99'], [sum(reached_goal1), sum(reached_goal2), sum(reached_goal3)])
plt.savefig('plots/efficiency.png', dpi=300)
plt.close()
