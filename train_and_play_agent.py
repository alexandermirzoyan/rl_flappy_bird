import gymnasium as gym
from env import FlappyBirdGame
from stable_baselines3 import DQN
import imageio
import pygame
import numpy as np
import os

# Path to save/load the model
model_path = "flappy_bird_dqn_model.zip"

# Check if the model exists
if os.path.exists(model_path):
    # Load the pre-trained model
    model = DQN.load(model_path)
    print("Model loaded successfully!")
else:
    # Train the model if it doesn't exist
    env = FlappyBirdGame()
    model = DQN("MlpPolicy", env, verbose=100)

    # Train the model
    timesteps = 100000
    model.learn(timesteps)

    # Save the model after training
    model.save(model_path)
    print("Model trained and saved successfully!")

# Create the environment
env = FlappyBirdGame()

# Reset the environment and get initial observation
obs, _ = env.reset()

frames = []

# Run the agent for 1000 steps or until done
for _ in range(1000):
    # Get the action from the trained model
    action, _ = model.predict(obs, deterministic=True)

    # Step in the environment with the predicted action
    obs, _, done, _, _ = env.step(action)

    # Render and capture the frame
    screen = env.render()
    frame = pygame.surfarray.array3d(screen)
    frame = frame.swapaxes(0, 1)

    frames.append(frame)

    # Stop if the game is done
    if done:
        break

# Save the frames as a GIF with a frame rate of 60 FPS
imageio.mimsave("flappy_bird_game.gif", frames, duration=1/60)

# Optionally: Display the rewards graph or further analysis
# rewards = np.mean(np.array(reward_callback.rewards).reshape(-1, 100), axis=1)
# plt.plot(rewards)
# plt.ylabel('some numbers')
# plt.show()
