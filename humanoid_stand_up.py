import gym
from gym.utils.play import play

#play(gym.make('ALE/Pong-v5', render_mode='rgb_array'))

#import gymnasium as gym
#env = gym.make('HumanoidStandup-v4')
#play(gym.make('HumanoidStandup-v4', render_mode='rgb_array'))
import pygame
mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
play(gym.make("CartPole-v1", render_mode='rgb_array'), keys_to_action=mapping)
#play(gym.make('CartPole-v1', render_mode='rgb_array'))
