from gym.envs.registration import register

register(id='gym-grid-v0',
         entry_point='gym_grid.envs:gridWorld'
         )