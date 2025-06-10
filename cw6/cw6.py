import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt


def q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)

    return q_table, rewards


def random_agent(env, episodes=500):
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)

    return rewards


def evaluate_agent(env, q_table, episodes=100):
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)

    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'all_rewards': rewards
    }


def plot_rewards(q_rewards, rand_rewards):
    plt.plot(q_rewards, label='Q-Learning')
    plt.plot(rand_rewards, label='Random Agent')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning vs Random Agent')
    plt.legend()
    plt.grid()
    plt.show()


env = gym.make("CliffWalking-v0")

q_table, q_rewards = q_learning(env)
rand_rewards = random_agent(env)
eval_stats = evaluate_agent(env, q_table)

print("Q-Learning Evaluation:")
print(f"Mean Reward: {eval_stats['mean']:.2f}")
print(f"Standard Deviation: {eval_stats['std']:.2f}")
print(f"Min: {eval_stats['min']}, Max: {eval_stats['max']}")

plot_rewards(q_rewards, rand_rewards)
