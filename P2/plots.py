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

diss1 = pickle.load(open(dir1 + 'PPO_model_dissonance.pkl', 'rb'))
diss2 = pickle.load(open(dir2 + 'PPO_model_dissonance.pkl', 'rb'))

entropy1 = pickle.load(open(dir1 + 'PPO_model_entropy.pkl', 'rb'))
entropy2 = pickle.load(open(dir2 + 'PPO_model_entropy.pkl', 'rb'))

vacuity1 = pickle.load(open(dir1 + 'PPO_model_vacuity.pkl', 'rb'))
vacuity2 = pickle.load(open(dir2 + 'PPO_model_vacuity.pkl', 'rb'))

reward1 = pickle.load(open(dir1 + 'PPO_model_rewards.pkl', 'rb'))
reward2 = pickle.load(open(dir2 + 'PPO_model_rewards.pkl', 'rb'))

episodes = range(1, len(diss1)+1)
plt.plot(episodes, diss1, label='Dissonance')
plt.plot(episodes, entropy1, label='Entropy')
plt.plot(episodes, vacuity1, label='Vacuity')
plt.plot(episodes, reward1, label='Rewards')
plt.legend()
plt.xlabel('Episodes')

plt.savefig('RL.png', dpi=300)
plt.close()

plt.plot(episodes, diss2, label='Dissonance')
plt.plot(episodes, entropy2, label='Entropy')
plt.plot(episodes, vacuity2, label='Vacuity')
plt.plot(episodes, reward2, label='Rewards')
plt.legend()
plt.xlabel('Episodes')

plt.savefig('RL_uncertainty_aware.png', dpi=300)
plt.close()
