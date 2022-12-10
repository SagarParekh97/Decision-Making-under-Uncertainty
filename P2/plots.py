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


dir1 = 'models/RL/'
dir2 = 'models/RLR_D/'
dir3 = 'models/RLR_H/'
dir4 = 'models/RLR_V/'
dir5 = 'models/RLG_D/'
dir6 = 'models/RLG_H/'
dir7 = 'models/RLG_V/'

diss1 = pickle.load(open(dir1 + 'PPO_model_dissonance.pkl', 'rb'))
diss2 = pickle.load(open(dir2 + 'PPO_model_dissonance.pkl', 'rb'))
diss3 = pickle.load(open(dir3 + 'PPO_model_dissonance.pkl', 'rb'))
diss4 = pickle.load(open(dir4 + 'PPO_model_dissonance.pkl', 'rb'))
diss5 = pickle.load(open(dir5 + 'PPO_model_dissonance.pkl', 'rb'))
diss6 = pickle.load(open(dir6 + 'PPO_model_dissonance.pkl', 'rb'))
diss7 = pickle.load(open(dir7 + 'PPO_model_dissonance.pkl', 'rb'))

entropy1 = pickle.load(open(dir1 + 'PPO_model_entropy.pkl', 'rb'))
entropy2 = pickle.load(open(dir2 + 'PPO_model_entropy.pkl', 'rb'))
entropy3 = pickle.load(open(dir3 + 'PPO_model_entropy.pkl', 'rb'))
entropy4 = pickle.load(open(dir4 + 'PPO_model_entropy.pkl', 'rb'))
entropy5 = pickle.load(open(dir5 + 'PPO_model_entropy.pkl', 'rb'))
entropy6 = pickle.load(open(dir6 + 'PPO_model_entropy.pkl', 'rb'))
entropy7 = pickle.load(open(dir7 + 'PPO_model_entropy.pkl', 'rb'))

vacuity1 = pickle.load(open(dir1 + 'PPO_model_vacuity.pkl', 'rb'))
vacuity2 = pickle.load(open(dir2 + 'PPO_model_vacuity.pkl', 'rb'))
vacuity3 = pickle.load(open(dir3 + 'PPO_model_vacuity.pkl', 'rb'))
vacuity4 = pickle.load(open(dir4 + 'PPO_model_vacuity.pkl', 'rb'))
vacuity5 = pickle.load(open(dir5 + 'PPO_model_vacuity.pkl', 'rb'))
vacuity6 = pickle.load(open(dir6 + 'PPO_model_vacuity.pkl', 'rb'))
vacuity7 = pickle.load(open(dir7 + 'PPO_model_vacuity.pkl', 'rb'))

reward1 = pickle.load(open(dir1 + 'PPO_model_rewards.pkl', 'rb'))
reward2 = pickle.load(open(dir2 + 'PPO_model_rewards.pkl', 'rb'))
reward3 = pickle.load(open(dir3 + 'PPO_model_rewards.pkl', 'rb'))
reward4 = pickle.load(open(dir4 + 'PPO_model_rewards.pkl', 'rb'))
reward5 = pickle.load(open(dir5 + 'PPO_model_rewards.pkl', 'rb'))
reward6 = pickle.load(open(dir6 + 'PPO_model_rewards.pkl', 'rb'))
reward7 = pickle.load(open(dir7 + 'PPO_model_rewards.pkl', 'rb'))

reached_goal1 = pickle.load(open(dir1 + 'PPO_model_reached_goal.pkl', 'rb'))
reached_goal2 = pickle.load(open(dir2 + 'PPO_model_reached_goal.pkl', 'rb'))
reached_goal3 = pickle.load(open(dir3 + 'PPO_model_reached_goal.pkl', 'rb'))
reached_goal4 = pickle.load(open(dir4 + 'PPO_model_reached_goal.pkl', 'rb'))
reached_goal5 = pickle.load(open(dir5 + 'PPO_model_reached_goal.pkl', 'rb'))
reached_goal6 = pickle.load(open(dir6 + 'PPO_model_reached_goal.pkl', 'rb'))
reached_goal7 = pickle.load(open(dir7 + 'PPO_model_reached_goal.pkl', 'rb'))

if not os.path.exists('plots/'):
    os.mkdir('plots/')

episodes1 = range(1, len(diss1)+1)
episodes2 = range(1, len(diss2)+1)
episodes3 = range(1, len(diss3)+1)
episodes4 = range(1, len(diss4)+1)
episodes5 = range(1, len(diss5)+1)
episodes6 = range(1, len(diss6)+1)
episodes7 = range(1, len(diss7)+1)
plt.plot(episodes1, diss1, label='RL')
plt.plot(episodes2, diss2, label='RLR_D')
plt.plot(episodes3, diss3, label='RLR_H')
plt.plot(episodes4, diss4, label='RLR_V')
plt.plot(episodes5, diss5, label='RLG_D')
plt.plot(episodes6, diss6, label='RLG_H')
plt.plot(episodes7, diss7, label='RLG_V')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Dissonance')
plt.savefig('plots/Dissonance.png', dpi=300)
plt.close()

plt.plot(episodes1, entropy1, label='RL')
plt.plot(episodes2, entropy2, label='RLR_D')
plt.plot(episodes3, entropy3, label='RLR_H')
plt.plot(episodes4, entropy4, label='RLR_V')
plt.plot(episodes5, entropy5, label='RLG_D')
plt.plot(episodes6, entropy6, label='RLG_H')
plt.plot(episodes7, entropy7, label='RLG_V')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Entropy')
plt.savefig('plots/Entropy.png', dpi=300)
plt.close()

plt.plot(episodes1, vacuity1, label='RL')
plt.plot(episodes2, vacuity2, label='RLR_D')
plt.plot(episodes3, vacuity3, label='RLR_H')
plt.plot(episodes4, vacuity4, label='RLR_V')
plt.plot(episodes5, vacuity5, label='RLG_D')
plt.plot(episodes6, vacuity6, label='RLG_H')
plt.plot(episodes7, vacuity7, label='RLG_V')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Vacuity')
plt.savefig('plots/Vacuity.png', dpi=300)
plt.close()

plt.plot(episodes1, EMA_filter(reward1), label='RL')
plt.plot(episodes2, EMA_filter(reward2), label='RLR_D')
plt.plot(episodes3, EMA_filter(reward3), label='RLR_H')
plt.plot(episodes4, EMA_filter(reward4), label='RLR_V')
plt.plot(episodes5, EMA_filter(reward5), label='RLG_D')
plt.plot(episodes6, EMA_filter(reward6), label='RLG_H')
plt.plot(episodes7, EMA_filter(reward7), label='RLG_H')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig('plots/Rewards.png', dpi=300)
plt.close()



plt.plot(episodes1, diss1, label='RL')
plt.plot(episodes2, diss2, label='RLR_D')
plt.plot(episodes3, diss3, label='RLR_H')
plt.plot(episodes4, diss4, label='RLR_V')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Dissonance')
plt.savefig('plots/Dissonance_RLR.png', dpi=300)
plt.close()

plt.plot(episodes1, entropy1, label='RL')
plt.plot(episodes2, entropy2, label='RLR_D')
plt.plot(episodes3, entropy3, label='RLR_H')
plt.plot(episodes4, entropy4, label='RLR_V')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Entropy')
plt.savefig('plots/Entropy_RLR.png', dpi=300)
plt.close()

plt.plot(episodes1, vacuity1, label='RL')
plt.plot(episodes2, vacuity2, label='RLR_D')
plt.plot(episodes3, vacuity3, label='RLR_H')
plt.plot(episodes4, vacuity4, label='RLR_V')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Vacuity')
plt.savefig('plots/Vacuity_RLR.png', dpi=300)
plt.close()

plt.plot(episodes1, EMA_filter(reward1), label='RL')
plt.plot(episodes2, EMA_filter(reward2), label='RLR_D')
plt.plot(episodes3, EMA_filter(reward3), label='RLR_H')
plt.plot(episodes4, EMA_filter(reward4), label='RLR_V')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig('plots/Rewards_RLR.png', dpi=300)
plt.close()





plt.plot(episodes1, diss1, label='RL')
plt.plot(episodes5, diss5, label='RLG_D')
plt.plot(episodes6, diss6, label='RLG_H')
plt.plot(episodes7, diss7, label='RLG_V')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Dissonance')
plt.savefig('plots/Dissonance_RLG.png', dpi=300)
plt.close()

plt.plot(episodes1, entropy1, label='RL')
plt.plot(episodes5, entropy5, label='RLG_D')
plt.plot(episodes6, entropy6, label='RLG_H')
plt.plot(episodes7, entropy7, label='RLG_V')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Entropy')
plt.savefig('plots/Entropy_RLG.png', dpi=300)
plt.close()

plt.plot(episodes1, vacuity1, label='RL')
plt.plot(episodes5, vacuity5, label='RLG_D')
plt.plot(episodes6, vacuity6, label='RLG_H')
plt.plot(episodes7, vacuity7, label='RLG_V')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Vacuity')
plt.savefig('plots/Vacuity_RLG.png', dpi=300)
plt.close()

plt.plot(episodes1, EMA_filter(reward1), label='RL')
plt.plot(episodes5, EMA_filter(reward5), label='RLG_D')
plt.plot(episodes6, EMA_filter(reward6), label='RLG_H')
plt.plot(episodes7, EMA_filter(reward7), label='RLG_V')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig('plots/Rewards_RLG.png', dpi=300)
plt.close()


