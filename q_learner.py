#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import gym
import numpy as np


'''
Variables needed for Q learning
'''
MAX_NUM_EPISODES = 100
STEPS_PER_EPISODE = 300 #  This is specific to MountainCar. May change with env
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.1  # Learning rate
GAMMA = 0.98  # Discount factor
# NUM_DISCRETE_BINS = 30  # Number of bins to Discretize each observation dim

def train(agent, env):
    best_reward = -float('inf')
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = env.reset()
        total_reward = 0.0
        while not done:
            # 1. Start with an empty table mapping states to values of actions
            str_obs = str(obs)
            if str_obs not in agent.Q:
                agent.Q[str_obs] = dict.fromkeys(range(env.action_space.n+1), 0)
            action = agent.get_action(str_obs)
            next_obs, reward, done, info = env.step(action)
            str_next_obs = str(next_obs)

            if str_next_obs not in agent.Q:
                agent.Q[str_next_obs] = dict.fromkeys(range(env.action_space.n+1),0)
            agent.learn(str_obs, action, reward, str_next_obs)
            obs = next_obs
            total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward
        print("Final State ", obs)
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode,
                                     total_reward, best_reward, agent.epsilon))
    # Return the trained policy
    max_v=-200
    max_k=0
    for key in agent.Q:
        print(key)
        for k,v in agent.Q[key].items():
            if v > max_v:
                max_v = v
                max_k = k
    # value, key = max((v,k) for inner_q in agent.Q for k, v  in inner_q.items())
    print(max_v, max_k)
    return agent.Q

def test(agent, env, policy):
    done = False
    obs = env.reset()
    total_reward = 0.0
    while not done:
        value, key = max((v,k) for k, v  in policy[str(obs)].items())
        # print(value, key)
        action = key
        # if obs not in agent.Q:
        #     agent.Q[str(obs)] = dict.fromkeys(range(env.action_space.n+1), 0)
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        total_reward += reward
        print(reward)
    return total_reward

class Q_Learner(object):
    def __init__(self, env, size):
        self.obs_shape = env.observation_space.shape[0]
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.action_shape = env.action_space.n
        self.size = size*size*(size-1) + ((size-1)*size + (size-1))
        self.Q = {}
        self.alpha = ALPHA  # Learning rate
        self.gamma = GAMMA  # Discount factor
        self.epsilon = 1.0

    def discretize(self, obs):
        return tuple(((obs - self.obs_low) / self.bin_width).astype(int))

    def get_action(self, obs):
        # Epsilon-Greedy action selection
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:
            value, key = max((v,k) for k, v in self.Q[obs].items())
            print(value, key)
            return key
        else:  # Choose a random action
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs):
        value, key = max((v,k) for k,v in self.Q[next_obs].items())
        # print(f"Value {value}")
        td_target = reward + self.gamma * value
        td_error = td_target - self.Q[obs][action]
        self.Q[obs][action] += self.alpha * td_error
        # print("obs: {}, reward: {}, action: {}, value: {}".format(obs, reward, action, self.alpha*td_error))