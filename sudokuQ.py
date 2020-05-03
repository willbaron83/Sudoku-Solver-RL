#Q Learning Algorithm Sudoku
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:12:34 2020

@author: epeatfield
"""
import gym
from gym import spaces
import numpy as np
from newsudoku import SudokuEnv
import random

max_episodes = 1
steps_per_episode = 200
epsilon_min = 0.005
max_num_steps = max_episodes * steps_per_episode
epsilon_decay = 500 * epsilon_min / max_num_steps
alpha = 0.05
gamma = 0.98
num_discrete_bins = 30
epsilon = 1.0

class Q_Learner_Sudoku(object):
    def __init__(self, env):
        self.obs_shape = spaces.Box(low=1, high=9, shape=(3, 3))
        self.obs_bins = num_discrete_bins
        self.action_space = env.action_space
        # self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1, self.action_shape))
        self.Q = {}
        self.RETURNS = {}
        self.A = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_action(self, obs):
        if self.epsilon > epsilon_min:
            self.epsilon -= epsilon_decay
        tp = np.random.random()
        # print("Random Num ", tp, " Epsilon ", self.epsilon)
        if tp > self.epsilon:
            # print("Q Obs = ", self.Q[obs])
            Q_obs = self.Q[obs]
            max_val = 0
            max_action = ''
            if Q_obs == {}:
                act = self.action_space.sample()
                list_act = list(act)
                list_act.pop()
                print("List ", list_act)
                tup_act = tuple(list_act)
                ax = str(tup_act)
                self.Q[obs][ax] = {0: 0, 1: 0, 2: 0}
            for action in Q_obs:
                print("ACTION ", action, " ITEMS ", Q_obs[action])
                inner_Q_obs = Q_obs[action]
                max_v = max(inner_Q_obs, key=inner_Q_obs.get)
                if max_v >= max_val:
                    max_val = max_v
                    max_action = action
            # print(max_val, max_action)
            # value, action_value = max(((v,k) for inner_q in Q_obs for k, v in inner_q.items()))
            #value = {0: 0, 1: 0, 2: 0}
            print("Value ", max_val, " Action ", max_action)
            temp = []
            for t in max_action.split(", "):
                num = int(t.replace("(", "").replace(")", ""))
                temp.append(num)
                if ")" in t:
                    temp.append(max_val)
                    a = tuple(temp)
                    print("Final Action ", a)
            # print("Q ", self.Q)
            return a
        else:
            print("Smaller")
            return self.action_space.sample()

    def learn(self, obs, action, reward, next_obs):
        print("next ", next_obs)
        td_target = reward + self.gamma * np.argmax(self.Q[next_obs])
        #action = '(2,1,2)'
        tmep = []
        for t in action.split(','):
            num = int(t.replace('(','').replace(')', ''))
            tmep.append(num)
        val = tmep.pop()
        a = str(tuple(tmep))
        if a not in agent.Q[obs]:
                agent.Q[obs][a] = {0: 0, 1: 0, 2: 0}
        td_error = td_target - self.Q[obs][a][val]
        self.Q[obs][a][val] += self.alpha * td_error


def train(agent, env):
    print("EHLLO")
    best_reward = -float('inf')
    for episode in range(max_episodes):
        done = False
        obs = env.reset()
        total = 0.0
        while not done:
            state = str(obs)
            if state not in agent.Q:
                agent.Q[state] = {}
                agent.RETURNS[state] = {}
                
            action = agent.get_action(state)
            action_str = str(action)
            agent.A[action_str] = action
            next_obs, reward, done, info = env.step(action)
            next_state = str(next_obs)
            if next_state not in agent.Q:
                agent.Q[next_state] = {}
            agent.learn(state, action_str, reward, next_state)
            obs = next_obs
            total += reward
        if total > best_reward:
            best_reward = total
    
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode,
                                                                   total, best_reward, agent.epsilon))
    return np.max(agent.Q)


def test(agent, env, policy):
    done = False
    obs = env.reset()
    total_reward = 0.0
    while not done:
        action = policy[obs]
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        total_reward += reward
    return total_reward


# if __name__ == "main":
env = SudokuEnv()
agent = Q_Learner_Sudoku(env)
print("HELLO! ")
learned_policy = train(agent, env)
print("LEARNED", learned_policy)

