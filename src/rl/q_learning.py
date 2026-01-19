import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_bins=(10, 10, 10), n_actions=3,
                 alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.05):
        
        self.state_bins = state_bins
        self.n_actions = n_actions
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.Q = np.zeros(state_bins + (n_actions,))

    def discretize(self, state, env):
        bx = int(state[0] / (env.width / self.state_bins[0]))
        by = int(state[1] / (env.height / self.state_bins[1]))
        px = int(state[2] / (env.width / self.state_bins[2]))

        return (
            np.clip(bx, 0, self.state_bins[0]-1),
            np.clip(by, 0, self.state_bins[1]-1),
            np.clip(px, 0, self.state_bins[2]-1)
        )

    def choose_action(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.Q[s])

    def update(self, s, a, r, s_next, done):
        best_next = np.max(self.Q[s_next])
        target = r + (self.gamma * best_next * (not done))
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
