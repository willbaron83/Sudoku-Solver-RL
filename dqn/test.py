#!/usr/bin/env/ python
"""
q_learner.py
An easy-to-follow script to train, test and evaluate a Q-learning agent on the Mountain Car
problem using the OpenAI Gym. |Praveen Palanisamy
# Chapter 5, Hands-on Intelligent Agents with OpenAI Gym, 2018
"""
import gym
import numpy as np
from sudokuenv import SudokuEnv

#MAX_NUM_EPISODES = 500
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


if __name__ == "__main__":
    env = SudokuEnv(3)
    agent = Q_Learner(env, 3)
    learned_policy = train(agent, env)
    # print(learned_policy.shape)
    # print(learned_policy)
    # Use the Gym Monitor wrapper to evalaute the agent and record video
    # gym_monitor_path = "./gym_monitor_output"
    # env = gym.wrappers.Monitor(env, gym_monitor_path, force=True)
    for _ in range(1000):
        test(agent, env, learned_policy)
    env.close()

