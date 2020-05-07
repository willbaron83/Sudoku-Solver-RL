'''
Adapted from Keras DQN Implementation
Resource: https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288

Modified to work with our Sudoku Environment.
Modified by: Emma Peatfield & William Baron
Modified for: CMPE 297 Final Project
'''

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sudokuenv import SudokuEnv
import json
# from scores.score_logger import ScoreLogger

GAMMA = 0.3
LEARNING_RATE = 0.1

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.005
EXPLORATION_DECAY = 0.8


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


def sudoku(n):
    # env = gym.make(ENV_NAME)
    env = SudokuEnv(n)
    details = {}
    # score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0        
    best_reward = -float('inf')
    while run < 100:
        run += 1
        state = env.reset()
        # print(state)
        # print(observation_space)
        state = np.reshape(state, [observation_space, observation_space])
        step = 0
        total_reward = 0.0
        while True:
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [observation_space, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            total_reward += reward
            # print(state)
            if terminal:
                if total_reward > best_reward:
                    best_reward = total_reward
                details[f"trial{run}"] = {"run": run, "steps": step, "totalreward": reward, "best_reward": best_reward, "exploration": dqn_solver.exploration_rate}  
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                # score_logger.add_score(step, run)
                break
            # print(details)
            dqn_solver.experience_replay()
    #save deatils tp a file
    with open(f"DataValue{n}.json", "w") as f:
        json.dump(details,f)

if __name__ == "__main__":
    # for n in range(3):
    sudoku(3)