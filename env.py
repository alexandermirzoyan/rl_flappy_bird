import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import random
import os

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
BIRD_WIDTH, BIRD_HEIGHT = 40, 40
TUBE_WIDTH = 60
GROUND_HEIGHT = 100
GROUND_Y = SCREEN_HEIGHT - GROUND_HEIGHT
FONT_SIZE = 24

YELLOW = (245, 236, 50)
GREEN = (14, 99, 9)
LIGHT_BLUE = (132, 196, 204)
BLACK = (0, 0, 0)

FPS = 60
BIRD_X = 50
TUBE_SPEED = 5
BIRD_GRAVITY_SPEED = 3
TUBE_EMPTY_SPACE = 175
TUBES_DISTANCE = 300

class FlappyBirdGame(gym.Env):
    def __init__(self):
        super(FlappyBirdGame, self).__init__()
        # self.state: bird_y, tube_x, tube_top_y, tube_bottom_y
        # self.reward: reward for passing the tube, penalties for hitting on tube
        # self.action: 0: do nothing, 1: jump

        self.action_space = spaces.Discrete(2)  # 0, 1
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 100, 350 + TUBE_EMPTY_SPACE, 0, 100, 350 + TUBE_EMPTY_SPACE]),
            high=np.array([GROUND_Y, SCREEN_WIDTH, 350, 700, SCREEN_WIDTH, 350, 700]),
            shape=(7,),
            dtype=np.float32
        )

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, FONT_SIZE)

        # Images setup
        self.bird_image = pygame.image.load('./images/flappybird.png')
        self.floor_image = pygame.image.load('./images/floor.png')
        self.background_image = pygame.image.load('./images/background.png')
        self.background_image = pygame.transform.scale(self.background_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.bird_y = (SCREEN_HEIGHT // 2) - (BIRD_HEIGHT // 2)
        self.tube_x = SCREEN_WIDTH

        self.score = 0

        self.is_jumping = False

        self.tubes = []

        self.state = np.array([self.bird_y, 0, 100, 350 + TUBE_EMPTY_SPACE, 0, 100, 350 + TUBE_EMPTY_SPACE], dtype=np.float32)

        return self.state, {}

    def bird_jumping(self, action):
        if action == 1:
            self.bird_y -= BIRD_GRAVITY_SPEED * 25
        else:
            self.bird_y += BIRD_GRAVITY_SPEED

    def check_bird_and_ground_collision(self):
        if self.bird_y + BIRD_HEIGHT >= GROUND_Y or self.bird_y + BIRD_HEIGHT <= 0:
            return True

        return False

    def check_bird_and_tube_collision(self):
        has_top_tube_collision = (self.bird_y <= self.tubes[0]['top_y']) and (BIRD_X + BIRD_WIDTH >= self.tubes[0]['x'])
        has_bottom_tube_collision = (self.bird_y + BIRD_HEIGHT >= (GROUND_Y - self.tubes[0]['bottom_y'])) and (BIRD_X + BIRD_WIDTH >= self.tubes[0]['x'])

        if has_top_tube_collision or has_bottom_tube_collision:
            return True

        return False

    def check_bird_passed_tube(self):
        if BIRD_X == self.tubes[0]['x']:
            return True

        return False

    def create_random_tubes(self):
        if len(self.tubes) < 2 :
            for index in range(2):
                random_number = random.randint(100, GROUND_Y // 2)
                top_y = random_number
                bottom_y = GROUND_Y - top_y - TUBE_EMPTY_SPACE
                x = SCREEN_WIDTH - TUBE_WIDTH
                if index == 1:
                    x += TUBES_DISTANCE

                self.tubes.append({ "x": x, "top_y": top_y, "bottom_y": bottom_y })

    def create_new_tube(self):
        if len(self.tubes) == 2:
            first_tube = self.tubes[0]
            if (first_tube['x'] + TUBE_WIDTH < 0):
                self.tubes.pop(0)
                random_number = random.randint(100, GROUND_Y // 2)
                top_y = random_number
                bottom_y = GROUND_Y - top_y - TUBE_EMPTY_SPACE
                x = SCREEN_WIDTH - TUBE_WIDTH
                self.tubes.append({ "x": x, "top_y": top_y, "bottom_y": bottom_y })

    def move_tubes(self):
        for tube in self.tubes:
            tube['x'] -= TUBE_SPEED

    def step(self, action):
        self.bird_jumping(action)
        reward = 0
        if action == 1:
            reward += 5

        self.create_random_tubes()
        self.move_tubes()

        self.create_new_tube()

        if self.check_bird_passed_tube():
            self.score += 1
            reward += 20

        if self.bird_y >= self.tubes[0]['top_y'] and self.bird_y + BIRD_HEIGHT <= GROUND_Y - self.tubes[0]['bottom_y']:
            reward += 20

        done = self.check_bird_and_ground_collision() or self.check_bird_and_tube_collision()

        if done:
            reward = -100

        self.state = np.array([
            self.bird_y,
            self.tubes[0]['x'],
            self.tubes[0]['top_y'],
            self.tubes[0]['bottom_y'],
            self.tubes[1]['x'],
            self.tubes[1]['top_y'],
            self.tubes[1]['bottom_y']
        ], dtype=np.float32)

        return self.state, reward, done, False, {}

    def render(self, mode="human"):
        # self.screen.fill(LIGHT_BLUE)
        self.screen.blit(self.background_image, (0, 0))

        # Drawing Floor
        pygame.draw.line(self.screen, BLACK, (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y))
        self.screen.blit(self.floor_image, (0, GROUND_Y))
        self.floor_image = pygame.transform.scale(self.floor_image, (SCREEN_WIDTH, GROUND_HEIGHT))

        # Drawing Bird
        pygame.draw.rect(self.screen, YELLOW, (BIRD_X, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT))
        self.screen.blit(self.bird_image, (BIRD_X, self.bird_y))
        self.bird_image = pygame.transform.scale(self.bird_image, (BIRD_WIDTH, BIRD_HEIGHT))

        # Drawing Tubes - top and bottom part separately
        for tube in self.tubes:
            pygame.draw.rect(self.screen, GREEN, (tube['x'], 0, TUBE_WIDTH, tube['top_y']))
            pygame.draw.rect(self.screen, GREEN, (tube['x'], GROUND_Y - tube['bottom_y'], TUBE_WIDTH, tube['bottom_y']))

        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(FPS)

        return self.screen

    def close(self):
        pygame.quit()