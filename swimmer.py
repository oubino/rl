import gym
from gym.utils.play import play
import pygame
import numpy as np

mapping = {(pygame.K_UP,): np.array([1.0, 1.0]), (pygame.K_DOWN,): np.array([-1.0, -1.0]), (pygame.K_LEFT,): np.array([1.0, -1.0]), (pygame.K_RIGHT,): np.array([-1.0, 1.0])}
play(gym.make('Swimmer-v4', render_mode='rgb_array'), keys_to_action=mapping, noop=np.array([0.0, 0.0]))

