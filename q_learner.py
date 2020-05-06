#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import gym
import numpy as np


'''
Variables needed for Q learning
'''
MAX_NUM_EPISODES = 1
STEPS_PER_EPISODE = 300 #  This is specific to MountainCar. May change with env
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.1  # Learning rate
GAMMA = 0.98  # Discount factor
NUM_DISCRETE_BINS = 30  # Number of bins to Discretize each observation dim

class Q_Learner(object):
    def __init__(self, env, size):
        self.obs_shape = env.observation_space.shape[0]
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS  # Number of bins to Discretize each observation dim
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = env.action_space.n
        print("ACTION SHAPE {}".format(self.action_shape))
        # Create a multi-dimensional array (aka. Table) to represent the
        # Q-values
        # self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1,
        #                    self.action_shape))  # (51 x 51 x 3)
        self.size = size*size*(size-1) + ((size-1)*size + (size-1))
        self.Q = {}
        self.alpha = ALPHA  # Learning rate
        self.gamma = GAMMA  # Discount factor
        self.epsilon = 1.0

    def discretize(self, obs):
        return tuple(((obs - self.obs_low) / self.bin_width).astype(int))

    def get_action(self, obs):
        # discretized_obs = self.discretize(obs)
        # Epsilon-Greedy action selection
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:
            # print("Q ", self.Q[obs].items())
            value, key = max((v,k) for k, v  in self.Q[obs].items())
            print(value, key)
            return key
        else:  # Choose a random action
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs):
        # discretized_obs = self.discretize(obs)
        # discretized_next_obs = self.discretize(next_obs)
        # print("print", self.Q[next_obs])
        value, key = max((v,k) for k,v in self.Q[next_obs].items())
        print(f"Value {value}")
        td_target = reward + self.gamma * value
        # if action not in self.Q[obs]:
        #     self.Q[obs][action] = 0
        td_error = td_target - self.Q[obs][action]
        self.Q[obs][action] += self.alpha * td_error
        print("obs: {}, reward: {}, action: {}, value: {}".format(obs, reward, action, self.alpha*td_error))