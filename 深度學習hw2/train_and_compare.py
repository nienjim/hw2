import os
import random
import numpy as np
import matplotlib.pyplot as plt


class CliffGridEnv:
    def __init__(self, rows=4, cols=12):
        self.rows = rows
        self.cols = cols
        self.start = (rows - 1, 0)
        self.goal = (rows - 1, cols - 1)
        # cliff positions: bottom row, columns 1..cols-2
        self.cliff = {(rows - 1, c) for c in range(1, cols - 1)}
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self._state_to_index(self.state)

    def step(self, action):
        # actions: 0=up,1=right,2=down,3=left
        r, c = self.state
        if action == 0:
            r = max(r - 1, 0)
        elif action == 1:
            c = min(c + 1, self.cols - 1)
        elif action == 2:
            r = min(r + 1, self.rows - 1)
        elif action == 3:
            c = max(c - 1, 0)

        new_state = (r, c)

        if new_state in self.cliff:
            reward = -100
            self.state = self.start
            done = False
            return self._state_to_index(self.state), reward, done, {"fell": True}

        self.state = new_state
        if self.state == self.goal:
            return self._state_to_index(self.state), 0, True, {}

        return self._state_to_index(self.state), -1, False, {}

    def _state_to_index(self, s):
        r, c = s
        return r * self.cols + c

    def n_states(self):
        return self.rows * self.cols

    def n_actions(self):
        return 4

    def index_to_state(self, idx):
        return (idx // self.cols, idx % self.cols)


def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(Q.shape[1])
    return int(np.argmax(Q[state]))


def q_learning(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    nS, nA = env.n_states(), env.n_actions()
    Q = np.zeros((nS, nA))
    rewards = []

    for ep in range(episodes):
        s = env.reset()
        done = False
        total_r = 0
        while not done:
            a = epsilon_greedy(Q, s, epsilon)
            s2, r, done, info = env.step(a)
            best_next = np.max(Q[s2])
            Q[s, a] += alpha * (r + gamma * best_next - Q[s, a])
            s = s2
            total_r += r
        rewards.append(total_r)

    return Q, rewards


def sarsa(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    nS, nA = env.n_states(), env.n_actions()
    Q = np.zeros((nS, nA))
    rewards = []

    for ep in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, epsilon)
        done = False
        total_r = 0
        while not done:
            s2, r, done, info = env.step(a)
            a2 = epsilon_greedy(Q, s2, epsilon)
            Q[s, a] += alpha * (r + gamma * Q[s2, a2] - Q[s, a])
            s, a = s2, a2
            total_r += r
        rewards.append(total_r)

    return Q, rewards


def plot_rewards(q_rewards, s_rewards, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    episodes = len(q_rewards)
    x = np.arange(episodes)
    plt.figure(figsize=(10, 5))
    plt.plot(x, q_rewards, label='Q-learning')
    plt.plot(x, s_rewards, label='SARSA')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Total Reward: Q-learning vs SARSA')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(out_dir, 'rewards.png')
    plt.savefig(path)
    plt.close()
    print('Saved rewards plot to', path)


def visualize_policy(Q, env, out_path):
    # follow greedy policy from start until goal
    s = env.reset()
    visited = []
    for _ in range(1000):
        visited.append(env.index_to_state(s))
        a = int(np.argmax(Q[s]))
        s2, r, done, info = env.step(a)
        s = s2
        if done:
            visited.append(env.index_to_state(s))
            break

    grid = np.zeros((env.rows, env.cols), dtype=int)
    for r, c in env.cliff:
        grid[r, c] = -1
    for i, (r, c) in enumerate(visited):
        grid[r, c] = 2 if (r, c) == env.goal else 3 if (r, c) == env.start else 1

    plt.figure(figsize=(8, 3))
    cmap = plt.get_cmap('tab20')
    plt.imshow(grid, cmap=cmap)
    plt.title('Policy path (greedy)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print('Saved policy visualization to', out_path)


def main():
    env = CliffGridEnv(rows=4, cols=12)
    episodes = 500
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1

    print('Training Q-learning...')
    Q_q, rewards_q = q_learning(env, episodes=episodes, alpha=alpha, gamma=gamma, epsilon=epsilon)

    print('Training SARSA...')
    env = CliffGridEnv(rows=4, cols=12)  # fresh env
    Q_s, rewards_s = sarsa(env, episodes=episodes, alpha=alpha, gamma=gamma, epsilon=epsilon)

    out_dir = 'outputs'
    plot_rewards(rewards_q, rewards_s, out_dir)
    visualize_policy(Q_q, CliffGridEnv(rows=4, cols=12), os.path.join(out_dir, 'policy_q.png'))
    visualize_policy(Q_s, CliffGridEnv(rows=4, cols=12), os.path.join(out_dir, 'policy_sarsa.png'))

    # Save numeric results
    np.save(os.path.join(out_dir, 'Q_q.npy'), Q_q)
    np.save(os.path.join(out_dir, 'Q_sarsa.npy'), Q_s)
    np.save(os.path.join(out_dir, 'rewards_q.npy'), np.array(rewards_q))
    np.save(os.path.join(out_dir, 'rewards_sarsa.npy'), np.array(rewards_s))

    print('Done. Outputs in', out_dir)


if __name__ == '__main__':
    main()
