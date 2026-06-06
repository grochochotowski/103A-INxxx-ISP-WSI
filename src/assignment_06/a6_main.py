import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ================== SETUP ==================
env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n

alpha = 0.9
gamma = 0.95
epsilon = 0.3
max_steps = 200

num_of_ind_runs = 25
num_episodes = 1000
averaged_reward = np.zeros(num_episodes)

# ================== Q-LEARNING ==================
for run in range(num_of_ind_runs):
    qtable = np.zeros((state_size, action_size))

    for episode in range(num_episodes):
        state, info = env.reset()
        reward = 0

        for step in range(max_steps):
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                best_actions = np.flatnonzero(qtable[state] == np.max(qtable[state]))
                action = np.random.choice(best_actions)

            next_state, step_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            qtable[state, action] = qtable[state, action] + alpha * (
                step_reward + gamma * np.max(qtable[next_state]) - qtable[state, action]
            )

            state = next_state
            reward = reward + step_reward

            if done:
                break

        averaged_reward[episode] = averaged_reward[episode] + reward

averaged_reward = averaged_reward / (num_of_ind_runs)
averaged_reward_base = averaged_reward


# ================== SECOND RUN FOR COMPARISON ==================
averaged_reward = np.zeros(num_episodes)

for run in range(num_of_ind_runs):
    qtable = np.zeros((state_size, action_size))

    for episode in range(num_episodes):
        state, info = env.reset()
        reward = 0

        for step in range(max_steps):
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                best_actions = np.flatnonzero(qtable[state] == np.max(qtable[state]))
                action = np.random.choice(best_actions)

            next_state, step_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            qtable[state, action] = qtable[state, action] + alpha * (
                step_reward + gamma * np.max(qtable[next_state]) - qtable[state, action]
            )

            state = next_state
            reward = reward + step_reward

            if done:
                break

        averaged_reward[episode] = averaged_reward[episode] + reward

averaged_reward = averaged_reward / (num_of_ind_runs)


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

env.close()