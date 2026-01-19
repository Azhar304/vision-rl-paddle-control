# Vision-RL Paddle Control

A simple **Reinforcement Learning project** where an agent learns to control a paddle and keep a ball in play using **Q-learning**.  
Built using Python and OpenCV for real-time visualization.

## Features

- Tabular Q-learning agent
- 1 Paddle + 1 Ball environment
- Real-time visualization using OpenCV
- Reward shaping for faster learning
- Epsilon-greedy policy with decay
- Floating reward display above the paddle
- Easily extendable to multi-ball or multi-paddle environments

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Azhar304/vision-rl-paddle-control.git
cd vision-rl-paddle-control/src
Create a virtual environment (recommended):


python -m venv venv
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
Install dependencies:


pip install -r requirements.txt
Usage
Train the Agent

python main_train.py
Trains the agent over multiple episodes.

Live visualization of paddle and ball.

Total reward and current reward displayed on the screen.

Evaluate the Agent

python main_eval.py
Runs the trained agent in the environment.