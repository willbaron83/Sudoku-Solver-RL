#!/usr/bin/env/ python
"""
q_learner.py
Adapted from Q-Learning class example

Modified for our Sudoku Environment. Q is now a dictionary.
Modified by: Emma Peatfield & William Baron
Modified for: CMPE 297 Final Project
"""
import gym
import numpy as np
from sudokuenv import SudokuEnv
import json


MAX_NUM_EPISODES = 100
STEPS_PER_EPISODE = 3000 #  This is specific to MountainCar. May change with env
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
        self.size = size*size*(size-1) + ((size-1)*size + (size-1))
        self.Q = {}
        self.alpha = ALPHA  # Learning rate
        self.gamma = GAMMA  # Discount factor
        self.epsilon = 1.0

    def discretize(self, obs):
        return tuple(((obs - self.obs_low) / self.bin_width).astype(int))

    def get_action(self, obs):
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:
            value, key = max((v,k) for k, v  in self.Q[obs].items())
            return key
        else:  # Choose a random action
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs):
        value, key = max((v,k) for k,v in self.Q[next_obs].items())
        td_target = reward + self.gamma * value
        td_error = td_target - self.Q[obs][action]
        self.Q[obs][action] += self.alpha * td_error


def train(agent, env):
    best_reward = -float('inf')
    details = {}
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = env.reset()
        total_reward = 0.0
        step = 0.0
        while not done:
            step += 1
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
        details[f"trial{episode}"] = {"run": episode, "steps": step, "totalreward": reward, "best_reward": best_reward}
        
    # Return the trained policy
    with open(f"QLearningData{env.n}.json", "w") as f:
        json.dump(details, f)
    max_v=-200
    max_k=0
    for key in agent.Q:
        print(key)
        for k,v in agent.Q[key].items():
            if v > max_v:
                max_v = v
                max_k = k
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
        # print(reward)
    return total_reward


if __name__ == "__main__":
    for n in range(3, 10):
        env = SudokuEnv(n)
        agent = Q_Learner(env, n)
        learned_policy = train(agent, env)
    # for _ in range(1000):
    #     test(agent, env, learned_policy)
    # env.close()
