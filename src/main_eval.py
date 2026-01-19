from env.paddle_env import PaddleEnv
from rl.q_learning import QLearningAgent
import numpy as np
import cv2

env = PaddleEnv()
agent = QLearningAgent()
agent.Q = np.load("q_table.npy")
agent.epsilon = 0.0

state = env.reset()

while True:
    s = agent.discretize(state, env)
    action = agent.choose_action(s)
    
    state, reward, done = env.step(action)
    env.render()

    if done:
        state = env.reset()

    if cv2.waitKey(1) == ord('q'):
        break

env.close()
