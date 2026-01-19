import cv2
import numpy as np
import random

class PaddleEnv:
    def __init__(self, width=500, height=600):
        self.width = width
        self.height = height

        # ball
        self.ball_radius = 10
        self.ball_speed = 5

        # paddle
        self.paddle_width = 100
        self.paddle_height = 15
        self.paddle_speed = 10

        # tracking for trajectory
        self.trajectory = []

        self.reset()

    def reset(self):
        self.ball_x = random.randint(50, self.width - 50)
        self.ball_y = 50

        self.vx = random.choice([-1, 1]) * self.ball_speed
        self.vy = self.ball_speed

        self.paddle_x = self.width // 2 - self.paddle_width // 2
        self.paddle_y = self.height - 50

        self.done = False
        self.trajectory = [(self.ball_x, self.ball_y)]  # reset trajectory

        # reward info
        self.current_reward = 0
        self.total_reward = 0
        self.episode = 0

        return self.get_state()

    def get_state(self):
        return np.array([self.ball_x, self.ball_y, self.paddle_x])

    def step(self, action):
        prev_dist = abs(self.ball_x - self.paddle_x)

        # ACTIONS
        if action == 0:
            self.paddle_x -= self.paddle_speed
        elif action == 1:
            self.paddle_x += self.paddle_speed

        self.paddle_x = np.clip(self.paddle_x, 0, self.width - self.paddle_width)

        # MOVE BALL
        self.ball_x += self.vx
        self.ball_y += self.vy
        self.trajectory.append((self.ball_x, self.ball_y))

        # WALL BOUNCE
        if self.ball_x <= self.ball_radius or self.ball_x >= self.width - self.ball_radius:
            self.vx *= -1
        if self.ball_y <= self.ball_radius:
            self.vy *= -1

        # reward shaping
        reward = 0.0
        new_dist = abs(self.ball_x - self.paddle_x)
        if new_dist < prev_dist:
            reward += 0.1
        else:
            reward -= 0.1
        reward += 0.01  # survival reward

        # check hit/miss
        if self.ball_y >= self.paddle_y - self.ball_radius:
            if self.paddle_x <= self.ball_x <= self.paddle_x + self.paddle_width:
                reward = +5
            else:
                reward = -5
            self.done = True

        self.current_reward = reward
        self.total_reward += reward

        return self.get_state(), reward, self.done

    def render(self):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Drawing ball trajectory orange
        for i in range(1, len(self.trajectory)):
            cv2.line(frame,
                     (int(self.trajectory[i-1][0]), int(self.trajectory[i-1][1])),
                     (int(self.trajectory[i][0]), int(self.trajectory[i][1])),
                     (0, 100, 255), 1)

        # Draw paddle
        cv2.rectangle(frame,
                      (self.paddle_x, self.paddle_y),
                      (self.paddle_x + self.paddle_width, self.paddle_y + self.paddle_height),
                      (255, 255, 255), -1)

        # Draw ball
        cv2.circle(frame, (int(self.ball_x), int(self.ball_y)), self.ball_radius, (0, 255, 255), -1)

        # displaying episode
        cv2.putText(frame, f"Episode: {self.episode}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("PaddleEnv", frame)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
