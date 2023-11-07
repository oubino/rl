import gym
from gym.utils.play import play
from gym.spaces import Box
import pygame
import numpy as np

env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')
pygame.init()
screen = pygame.display.set_mode((400, 400))
clock = pygame.time.Clock()

action_mapping = {
    pygame.K_LEFT: [-1.0],
    pygame.K_RIGHT: [1.0]
}

observation = env.reset()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    action = [0.0]  # Default action
    for key, value in action_mapping.items():
        if keys[key]:
            action = value

    env.render()
    observation, reward, done, info, x = env.step(action)

    if done:
        observation = env.reset()

    clock.tick(30)  # Limit frame rate

env.close()
pygame.quit()
