import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def EMA_filter(time_series, weight=0.99):
    last = time_series[0]
    smoothed = []
    for p in time_series:
        filt_val = weight * last + (1 - weight) * p
        smoothed.append(filt_val)
        last = filt_val

    return np.array(smoothed)



dir1 = 'models/RLR_D0.8/'
dir2 = 'models/RLR_D0.9/'
dir3 = 'models/RLR_D0.99/'
dir4 = 'models/RLR_Deps_0.2/'
dir5 = 'models/RLR_Deps_0.4/'
dir6 = 'models/RLR_Deps_0.1/'
dir7 = 'models/RLR_D20/'
dir8 = 'models/RLR_D30/'
dir9 = 'models/RLR_D40/'

diss1 = pickle.load(open(dir1 + 'PPO_model_dissonance.pkl', 'rb'))
diss2 = pickle.load(open(dir2 + 'PPO_model_dissonance.pkl', 'rb'))
diss3 = pickle.load(open(dir3 + 'PPO_model_dissonance.pkl', 'rb'))
diss4 = pickle.load(open(dir4 + 'PPO_model_dissonance.pkl', 'rb'))
diss5 = pickle.load(open(dir5 + 'PPO_model_dissonance.pkl', 'rb'))
diss6 = pickle.load(open(dir6 + 'PPO_model_dissonance.pkl', 'rb'))
diss7 = pickle.load(open(dir7 + 'PPO_model_dissonance.pkl', 'rb'))
diss8 = pickle.load(open(dir8 + 'PPO_model_dissonance.pkl', 'rb'))
diss9 = pickle.load(open(dir9 + 'PPO_model_dissonance.pkl', 'rb'))

entropy1 = pickle.load(open(dir1 + 'PPO_model_entropy.pkl', 'rb'))
entropy2 = pickle.load(open(dir2 + 'PPO_model_entropy.pkl', 'rb'))
entropy3 = pickle.load(open(dir3 + 'PPO_model_entropy.pkl', 'rb'))
entropy4 = pickle.load(open(dir4 + 'PPO_model_entropy.pkl', 'rb'))
entropy5 = pickle.load(open(dir5 + 'PPO_model_entropy.pkl', 'rb'))
entropy6 = pickle.load(open(dir6 + 'PPO_model_entropy.pkl', 'rb'))
entropy7 = pickle.load(open(dir7 + 'PPO_model_entropy.pkl', 'rb'))
entropy8 = pickle.load(open(dir8 + 'PPO_model_entropy.pkl', 'rb'))
entropy9 = pickle.load(open(dir9 + 'PPO_model_entropy.pkl', 'rb'))

vacuity1 = pickle.load(open(dir1 + 'PPO_model_vacuity.pkl', 'rb'))
vacuity2 = pickle.load(open(dir2 + 'PPO_model_vacuity.pkl', 'rb'))
vacuity3 = pickle.load(open(dir3 + 'PPO_model_vacuity.pkl', 'rb'))
vacuity4 = pickle.load(open(dir4 + 'PPO_model_vacuity.pkl', 'rb'))
vacuity5 = pickle.load(open(dir5 + 'PPO_model_vacuity.pkl', 'rb'))
vacuity6 = pickle.load(open(dir6 + 'PPO_model_vacuity.pkl', 'rb'))
vacuity7 = pickle.load(open(dir7 + 'PPO_model_vacuity.pkl', 'rb'))
vacuity8 = pickle.load(open(dir8 + 'PPO_model_vacuity.pkl', 'rb'))
vacuity9 = pickle.load(open(dir9 + 'PPO_model_vacuity.pkl', 'rb'))

reward1 = pickle.load(open(dir1 + 'PPO_model_rewards.pkl', 'rb'))
reward2 = pickle.load(open(dir2 + 'PPO_model_rewards.pkl', 'rb'))
reward3 = pickle.load(open(dir3 + 'PPO_model_rewards.pkl', 'rb'))
reward4 = pickle.load(open(dir4 + 'PPO_model_rewards.pkl', 'rb'))
reward5 = pickle.load(open(dir5 + 'PPO_model_rewards.pkl', 'rb'))
reward6 = pickle.load(open(dir6 + 'PPO_model_rewards.pkl', 'rb'))
reward7 = pickle.load(open(dir7 + 'PPO_model_rewards.pkl', 'rb'))
reward8 = pickle.load(open(dir8 + 'PPO_model_rewards.pkl', 'rb'))
reward9 = pickle.load(open(dir9 + 'PPO_model_rewards.pkl', 'rb'))

reached_goal1 = pickle.load(open(dir1 + 'PPO_model_reached_goal.pkl', 'rb'))
reached_goal2 = pickle.load(open(dir2 + 'PPO_model_reached_goal.pkl', 'rb'))
reached_goal3 = pickle.load(open(dir3 + 'PPO_model_reached_goal.pkl', 'rb'))
reached_goal4 = pickle.load(open(dir4 + 'PPO_model_reached_goal.pkl', 'rb'))
reached_goal5 = pickle.load(open(dir5 + 'PPO_model_reached_goal.pkl', 'rb'))
reached_goal6 = pickle.load(open(dir6 + 'PPO_model_reached_goal.pkl', 'rb'))
reached_goal7 = pickle.load(open(dir7 + 'PPO_model_reached_goal.pkl', 'rb'))
reached_goal8 = pickle.load(open(dir8 + 'PPO_model_reached_goal.pkl', 'rb'))
reached_goal9 = pickle.load(open(dir9 + 'PPO_model_reached_goal.pkl', 'rb'))

if not os.path.exists('plots/'):
    os.mkdir('plots/')

episodes1 = range(1, len(diss1)+1)
episodes2 = range(1, len(diss2)+1)
episodes3 = range(1, len(diss3)+1)
episodes4 = range(1, len(diss4)+1)
episodes5 = range(1, len(diss5)+1)
episodes6 = range(1, len(diss6)+1)
episodes7 = range(1, len(diss7)+1)
episodes8 = range(1, len(diss8)+1)
episodes9 = range(1, len(diss9)+1)

plt.plot(episodes1, diss1, label='RLD_gamma_0.8')
plt.plot(episodes2, diss2, label='RLD_gamma_0.9')
plt.plot(episodes3, diss3, label='RLD_gamma_0.99')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Dissonance')
plt.savefig('plots/Dissonance_gamma.png', dpi=300)
plt.close()

plt.plot(episodes1, entropy1, label='RLD_gamma_0.8')
plt.plot(episodes2, entropy2, label='RLD_gamma_0.9')
plt.plot(episodes3, entropy3, label='RLD_gamma_0.99')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Entropy')
plt.savefig('plots/Entropy_gamma.png', dpi=300)
plt.close()

plt.plot(episodes1, vacuity1, label='RLD_gamma_0.8')
plt.plot(episodes2, vacuity2, label='RLD_gamma_0.9')
plt.plot(episodes3, vacuity3, label='RLD_gamma_0.99')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Vacuity')
plt.savefig('plots/Vacuity_gamma.png', dpi=300)
plt.close()

plt.plot(episodes1, EMA_filter(reward1), label='RLD_gamma_0.8')
plt.plot(episodes2, EMA_filter(reward2), label='RLD_gamma_0.9')
plt.plot(episodes3, EMA_filter(reward3), label='RLD_gamma_0.99')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig('plots/Rewards_gamma.png', dpi=300)
plt.close()









plt.plot(episodes4, diss4, label='RLD_eps_0.8')
plt.plot(episodes5, diss5, label='RLD_eps_0.9')
plt.plot(episodes6, diss6, label='RLD_eps_0.99')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Dissonance')
plt.savefig('plots/Dissonance_eps.png', dpi=300)
plt.close()

plt.plot(episodes4, entropy4, label='RLD_eps_0.8')
plt.plot(episodes5, entropy5, label='RLD_eps_0.9')
plt.plot(episodes6, entropy6, label='RLD_eps_0.99')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Entropy')
plt.savefig('plots/Entropy_eps.png', dpi=300)
plt.close()

plt.plot(episodes4, vacuity4, label='RLD_eps_0.8')
plt.plot(episodes5, vacuity5, label='RLD_eps_0.9')
plt.plot(episodes6, vacuity6, label='RLD_eps_0.99')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Vacuity')
plt.savefig('plots/Vacuity_eps.png', dpi=300)
plt.close()

plt.plot(episodes4, EMA_filter(reward4), label='RLD_eps_0.8')
plt.plot(episodes5, EMA_filter(reward5), label='RLD_eps_0.9')
plt.plot(episodes6, EMA_filter(reward6), label='RLD_eps_0.99')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig('plots/Rewards_eps.png', dpi=300)
plt.close()







plt.plot(episodes7, diss7, label='RLD_20')
plt.plot(episodes8, diss8, label='RLD_30')
plt.plot(episodes9, diss9, label='RLD_40')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Dissonance')
plt.savefig('plots/Dissonance_goalRegion.png', dpi=300)
plt.close()

plt.plot(episodes7, entropy7, label='RLD_20')
plt.plot(episodes8, entropy8, label='RLD_30')
plt.plot(episodes9, entropy9, label='RLD_40')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Entropy')
plt.savefig('plots/Entropy_goalRegion.png', dpi=300)
plt.close()

plt.plot(episodes7, vacuity7, label='RLD_20')
plt.plot(episodes8, vacuity8, label='RLD_30')
plt.plot(episodes9, vacuity9, label='RLD_40')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Vacuity')
plt.savefig('plots/Vacuity_goalRegion.png', dpi=300)
plt.close()

plt.plot(episodes7, EMA_filter(reward7), label='RLD_20')
plt.plot(episodes8, EMA_filter(reward8), label='RLD_30')
plt.plot(episodes9, EMA_filter(reward9), label='RLD_40')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig('plots/Rewards_goalRegion.png', dpi=300)
plt.close()