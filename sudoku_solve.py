#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import gym
import numpy as np
from sudoku_gym_env import SudokuEnv
from q_learner import Q_Learner
from q_learner import train
from q_learner import test
from dqn_solver import DQNSolver

# from scores.score_logger import ScoreLogger

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
    # dqn_sudoku()
    q_learning_sudoku()