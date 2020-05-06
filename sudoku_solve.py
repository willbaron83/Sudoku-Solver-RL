#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sudoku_gym_env import SudokuEnv
from q_learner import Q_Learner
from dqn_solver import DQNSolver


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
# from scores.score_logger import ScoreLogger

'''
Variables needed for DQN
'''
GAMMA = 0.7
LEARNING_RATE = 0.1
MEMORY_SIZE = 1000000
BATCH_SIZE = 20
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.8

def train(agent, env):
    best_reward = -float('inf')
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = env.reset()
        total_reward = 0.0
        while not done:
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


def dqn_sudoku():
    env = SudokuEnv(9)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        state = env.reset()
        print(state)
        print(observation_space)
        state = np.reshape(state, [observation_space, observation_space])
        step = 0
        while True:
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [observation_space, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            print(state)
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                # score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()

def q_learning_sudoku():
    env = SudokuEnv(9)
    agent = Q_Learner(env, 9)
    learned_policy = train(agent, env)
    # print(learned_policy.shape)
    print(learned_policy)
    # Use the Gym Monitor wrapper to evaluate the agent and record video
    # gym_monitor_path = "./gym_monitor_output"
    # env = gym.wrappers.Monitor(env, gym_monitor_path, force=True)
    for _ in range(1000):
        test(agent, env, learned_policy)
    env.close()

if __name__ == "__main__":
    dqn_sudoku()
    # q_learning_sudoku()