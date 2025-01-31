import gymnasium as gym
from env import FlappyBirdGame

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import imageio
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt

import pygame

# IGNORE: WAY TO LOG AGENT LEARNING
from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super(RewardLoggerCallback, self).__init__()
        self.rewards = []

    def _on_step(self) -> bool:
        # Record the reward of the current step
        # print("reward :: ", self.locals["rewards"][0])
        self.rewards.append(self.locals["rewards"][0])
        return True

env = FlappyBirdGame()

model = DQN("MlpPolicy", env, verbose=100)

timesteps = 100000
reward_callback = RewardLoggerCallback()
model.learn(timesteps, callback=reward_callback)

env = FlappyBirdGame()
obs, _ = env.reset()

frames = []

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = env.step(action)

    screen = env.render()
    frame = pygame.surfarray.array3d(screen)
    frame = frame.swapaxes(0, 1)

    frames.append(frame)

    if done:
        break

imageio.mimsave("flappy_bird_game.gif", frames, duration=1/30)


# rewards are stored in reward_callback.rewards
# print(len(reward_callback.rewards))
# print(np.array(reward_callback.rewards).reshape(-1, 100))
reward = np.mean(np.array(reward_callback.rewards).reshape(-1, 100), axis=1)
# print(reward)

plt.plot(reward)
plt.ylabel('some numbers')
plt.show()