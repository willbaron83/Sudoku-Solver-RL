#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:58:02 2020

@author: epeatfield
"""

import numpy as np
from newsudoku import SudokuEnv
import gym
import random
import sys

episodes = 1
env = SudokuEnv()
episode_rewards = [0]*episodes

for episode_i in range(episodes):
    sys.stdout.write("\r" + str(episode_i+1) + "/" + str(episodes))
    print("Episode ", episode_i)
    sys.stdout.flush()
    obs = env.reset()
    done = False
    print(obs)
    while not done:
        print("OBS ", obs)
        action = env.action_space.sample()
        print("ACTION ", action)
        
        obs, reward, done, _ = env.step(action)
        episode_rewards[episode_i] += reward
    print("DONE! ", obs)