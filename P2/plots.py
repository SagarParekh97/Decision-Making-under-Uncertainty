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


dir1 = 'models/50_dynamic/'
dir2 = 'models/50_dynamic_uncertainty_aware/'
dir3 = 'models/results_dissonance/'

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

episodes1 = range(1, len(diss1)+1)
episodes2 = range(1, len(diss2)+1)
episodes3 = range(1, len(diss3)+1)
plt.plot(episodes1, diss1, label='RL')
plt.plot(episodes2, diss2, label='RL-H')
plt.plot(episodes3, diss3, label='RL-d')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Dissonance')
plt.savefig('Dissonance.png', dpi=300)
plt.close()

plt.plot(episodes1, entropy1, label='RL')
plt.plot(episodes2, entropy2, label='RL-H')
plt.plot(episodes3, entropy3, label='RL-d')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Entropy')
plt.savefig('Entropy.png', dpi=300)
plt.close()

plt.plot(episodes1, vacuity1, label='RL')
plt.plot(episodes2, vacuity2, label='RL-H')
plt.plot(episodes3, vacuity3, label='RL-d')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Vacuity')
plt.savefig('Vacuity.png', dpi=300)
plt.close()

plt.plot(episodes1, EMA_filter(reward1), label='RL')
plt.plot(episodes2, EMA_filter(reward2), label='RL-H')
plt.plot(episodes3, EMA_filter(reward3), label='RL-d')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig('Rewards.png', dpi=300)
plt.close()

plt.bar(['RL', 'RL-H', 'RL-d'], [sum(reached_goal1), sum(reached_goal2), sum(reached_goal3)])
plt.savefig('efficiency.png', dpi=300)
plt.close()
