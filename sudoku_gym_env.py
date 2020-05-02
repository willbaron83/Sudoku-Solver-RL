#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:12:34 2020

@author: epeatfield
"""
import gym
from gym import spaces
from sudoku_solve import Sudoku_Solve
import numpy as np
import gym_sudoku

max_episodes = 5
steps_per_episode = 20
epsilon_min = 0.005
max_num_steps = max_episodes * steps_per_episode
epsilon_decay = 500 * epsilon_min / max_num_steps
alpha = 0.05
gamma = 0.98
num_discrete_bins = 30


class Q_Learner_Sudoku():
    def __init__(self):
        self.obs_shape = spaces.Box(low=1, high=9, shape=(9, 9))
        self.obs_bins = num_discrete_bins
        self.action_shape = spaces.Tuple((spaces.Discrete(9), spaces.Discrete(9), spaces.Discrete(9)))

        self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1, self.action_shape))

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0

    def get_action(self, obs):
        if self.epsilon > epsilon_min:
            self.epsilon -= epsilon_decay
        if np.random.random() > self.epsilon:
            return np.argmax(selfQ[obs])
        else:
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs):
        td_target = reward + self.gamma * np.max(self.Q[next_obs])
        td_error = td_target - self.Q[obs][action]
        self.Q[obs][action] += self.alpha * td_error


def train(agent):
    best_reward = -float('inf')
    for episode in range(max_episodes):
        done = False
        obs = Sudoku_Solve.reset()
        total = 0.0
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done, info = Sudoku_Solve.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total += reward
        if total > best_reward:
            best_reward = total
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode,
                                                                   total_reward, best_reward, agent.epsilon))


def test(agent, policy):
    done = False
    obs = Sudoku_Solve.reset()
    total_reward = 0.0
    while not done:
        action = policy[obs]
        next_obs, reward, done, info = Sudoku_Solve.step(action)
        obs = next_obs
        total_reward += reward
    return total_reward


if __name__ == "main":
    env = gym.make('Sudoku-v0')
    agent = Q_Learner_Sudoku(env)
    # agent = Q_Learner_Sudoku()
    learned_policy = train(agent)
    print(learned_policy)


