import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# ================== SETUP ==================
alpha = 0.9
gamma = 0.95
epsilon = 0.3
max_steps = 200
num_of_ind_runs = 25


# ================== ACTION CHOICE ==================
def choose_action(env, qtable, state):
    if np.random.random() < epsilon:
        return env.action_space.sample()

    best_actions = np.flatnonzero(qtable[state] == np.max(qtable[state]))
    return np.random.choice(best_actions)


# ================== REWARD ==================
def get_reward(reward_type, state, next_state, step_reward, done):
    q_reward = step_reward

    if reward_type == "custom_1":
        if step_reward == 1:
            q_reward = 100
        elif done and step_reward == 0:
            q_reward = -1
        elif state == next_state:
            q_reward = -0.1
        else:
            q_reward = -0.01

    if reward_type == "custom_2":
        if step_reward == 1:
            q_reward = 10
        elif done and step_reward == 0:
            q_reward = -5
        elif state == next_state:
            q_reward = -0.5
        else:
            q_reward = -0.02

    return q_reward


# ================== Q-LEARNING ==================
def run_q_learning(is_slippery=False, num_episodes=1000, reward_type="base"):
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=is_slippery)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    averaged_reward = np.zeros(num_episodes)

    for run in range(num_of_ind_runs):
        qtable = np.zeros((state_size, action_size))

        for episode in range(num_episodes):
            state, info = env.reset()
            reward = 0

            for step in range(max_steps):
                action = choose_action(env, qtable, state)

                next_state, step_reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                q_reward = get_reward(reward_type, state, next_state, step_reward, done)

                if done:
                    target = q_reward
                else:
                    target = q_reward + gamma * np.max(qtable[next_state])

                qtable[state, action] = qtable[state, action] + alpha * (
                        target - qtable[state, action]
                )

                state = next_state
                reward = reward + step_reward

                if done:
                    break

            averaged_reward[episode] = averaged_reward[episode] + reward

    averaged_reward = averaged_reward / (num_of_ind_runs)

    env.close()
    return averaged_reward


# ================== PLOTS ==================
def draw_plot(averaged_reward_base, averaged_reward):
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


# ================== TESTS ==================
def run_test(is_slippery=False, num_episodes=1000):
    averaged_reward_base = run_q_learning(is_slippery, num_episodes, "base")

    averaged_reward = run_q_learning(is_slippery, num_episodes, "custom_1")
    print("custom_1 mean:", np.mean(averaged_reward))
    draw_plot(averaged_reward_base, averaged_reward)

    averaged_reward = run_q_learning(is_slippery, num_episodes, "custom_2")
    print("custom_2 mean:", np.mean(averaged_reward))
    draw_plot(averaged_reward_base, averaged_reward)


# ================== RUN ==================
run_test(is_slippery=False, num_episodes=1000)
run_test(is_slippery=True, num_episodes=10000)