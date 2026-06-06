import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ================== SET UP ==================
env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n


num_of_ind_runs = 25
num_episodes = 1000
averaged_reward = np.zeros(num_episodes)

# ================== IDK ==================
for run in range(num_of_ind_runs):
    qtable = np.zeros((state_size, action_size))
    ...
    for episode in range(num_episodes):
        ...
        averaged_reward[episode] = averaged_reward[episode] + reward
        ...

averaged_reward = averaged_reward / (num_of_ind_runs)
averaged_reward_base = averaged_reward


# ================== PLOTS ==================
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot(averaged_reward_base, 'r')
plt.plot(averaged_reward, 'b')
plt.show()