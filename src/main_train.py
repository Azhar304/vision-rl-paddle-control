from env.paddle_env import PaddleEnv
from rl.q_learning import QLearningAgent
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2

env = PaddleEnv()
agent = QLearningAgent()

episodes = 300
render_every = 5  # for rendering we can change this as we like
speed = 0.01

reward_history = []

for ep in range(episodes):
    state = env.reset()
    env.episode = ep + 1
    total_reward = 0

    while True:
        s = agent.discretize(state, env)
        action = agent.choose_action(s)
        next_state, reward, done = env.step(action)

        s_next = agent.discretize(next_state, env)
        agent.update(s, action, reward, s_next, done)

        total_reward += reward
        state = next_state

        if ep % render_every == 0:
            env.render()
            time.sleep(speed)

        if done:
            agent.decay()
            reward_history.append(total_reward)
            break

    print(f"Episode {ep+1}/{episodes}, Reward={total_reward:.2f}, Epsilon={agent.epsilon:.2f}")

env.close()
np.save("q_table.npy", agent.Q)


reward_history = np.array(reward_history)

# moving average smoothing
def moving_average(x, w=20):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode='valid')

smoothed = moving_average(reward_history, w=20)

plt.figure(figsize=(12,6))
plt.plot(reward_history, color='skyblue', alpha=0.4, label='Raw Reward')
plt.plot(np.arange(len(smoothed)) + 19, smoothed, color='blue', label='Smoothed Reward (MA=20)')
plt.xlabel("Episodes", fontsize=14)
plt.ylabel("Total Reward", fontsize=14)
plt.title("RL Agent Training Reward Curve", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)

# highlighting the max reward achieved
max_reward = np.max(reward_history)
max_idx = np.argmax(reward_history)
plt.scatter(max_idx, max_reward, color='red', label=f'Max Reward: {max_reward:.1f}', zorder=5)
plt.legend(fontsize=12)

plt.show()

