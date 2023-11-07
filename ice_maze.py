import gym
from gym.utils.play import play
import pygame


# custom map 
desc=["SFFF", "FHFH", "FFFH", "HFFG"]

mapping = {(pygame.K_a,): 0, (pygame.K_s,): 1, (pygame.K_d,): 2, (pygame.K_f,): 3}
play(gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='rgb_array'), keys_to_action=mapping)
