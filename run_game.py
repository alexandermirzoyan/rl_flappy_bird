from env import FlappyBirdGame
import pygame

env = FlappyBirdGame()
obs, _ = env.reset()

done = False
while not done:
    env.render()

    action = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1
    
    _, _, done, _, _ = env.step(action)

env.render()
env.close()