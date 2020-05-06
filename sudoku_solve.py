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

# from scores.score_logger import ScoreLogger
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


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

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
    # env = gym.make(ENV_NAME)
    env = SudokuEnv(9)
    # score_logger = ScoreLogger(ENV_NAME)
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


if __name__ == "__main__":
    dqn_sudoku()