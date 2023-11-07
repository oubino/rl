import gym
from gym.utils.play import play
import pygame

mapping = {(pygame.K_UP,): 2, (pygame.K_DOWN,): 3}
play(gym.make('Pong-v4', difficulty=0, mode=0, render_mode='rgb_array'), keys_to_action=mapping)
# play(gym.make('ALE/Pong-v5', difficulty=3, mode=0, render_mode='rgb_array'), keys_to_action=mapping)

