import os
import sys

from copy import copy
from six import StringIO
from pprint import pprint

import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np

resolved = 0
unfinished = 1
error = 2

def checkSolution(grid):
    N = len(grid)
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 0:
                return unfinished
            n = N//3
            iOffset = i//n*n
            jOffset = j//n*n
            square = grid[ iOffset:iOffset + n, jOffset:jOffset + n].flatten()

            uniqueInRow = countItem(grid[i], grid[i, j]) == 1
            uniqueInCol = countItem(grid[:,j:j+1].flatten(), grid[i,j]) == 1
            uniqueInSquare = countItem(square, grid[i,j]) == 1

            if not (uniqueInRow and uniqueInCol and uniqueInSquare):
                return error
    return resolved

def countItem(vector, item):
    count = 0
    for item2 in vector:
        if item2 == item: count += 1
    return count

def getSolutions(grid, stopAt=1, i=-1, j=-1, omit=-1):
    N = len(grid)
    check = checkSolution(grid)

    if check == resolved:
        return np.array([grid], dtype=int)
    if check == error:
        return np.empty(shape=(0,N,N), dtype=int)

    if i == -1:
        for i in range(N):
            for j in range(N):
                if grid[i,j] == 0: break
            if grid[i, j] == 0: break

    values = np.arange(1, N+1)
    np.random.shuffle(values)

    solutions = np.empty(shape=(0,N,N), dtype=int)
    for value in values:
        if omit == value: continue
        cGrid = np.copy(grid)
        cGrid[i, j] = value
        subSolutions = getSolutions(cGrid, stopAt=stopAt-len(solutions))
        solutions = np.concatenate((solutions, subSolutions))
        if len(solutions) >= stopAt:
            return solutions
    return solutions



class SudokuEnv(gym.Env):
    last_action = None

    def __init__(self, n):
        self.n = n
        self.val_limit = self.n - 1
        self.nxn = self.n*self.n
        self.observation_space = spaces.Box(1, self.n, shape=(self.n, self.n))
        #Action = nxn*value + (row*n + col), n = 9 here
        self.action_space = spaces.Discrete(self.nxn*self.val_limit + (self.val_limit*self.n + self.val_limit))
        self.grid = []
        self.original_indices_row = []
        self.original_indices_col = []
        self.base = getSolutions(np.zeros(shape=(self.n, self.n)))[0]

        N = len(self.base)
        positions = []
        for i in range(N):
            for j in range(N):
                positions.append((i, j))
        np.random.shuffle(positions)

        count = 0
        for i, j in positions:
            if count > 1:
                break
            oldValue = self.base[i,j]
            self.base[i,j] = 0
            solutions = getSolutions(self.base, stopAt=2, i=i, j=j, omit=oldValue)
            if len(solutions) == 0:
                count += 1
            else:
                self.base[i,j] = oldValue

    def step(self, action):
        """
        """
        if self.last_action != None and self.last_action == action:
            return np.copy(self.grid), -0.5, False, None
        self.last_action = action
        oldGrid = np.copy(self.grid)

        square = action%self.nxn
        col = square%self.n
        row = (square-col)//self.n
        val = action//self.nxn + 1

        for i in range(len(self.original_indices_row)):
            if col == self.original_indices_col[i] and row == self.original_indices_row[i]:
                print(f"ORIGINAL FILL Row: {row} Column: {col} Value: {val}")
                return np.copy(self.grid), -1, False, None
        
        if self.grid[row, col] == val:
            print("Already there")
            return np.copy(self.grid), -1, False, None
        
        self.grid[row, col] = val

        stats = checkSolution(self.grid)
        if stats == resolved:
            return np.copy(self.grid), 1, True, None
        elif stats == unfinished:
            return np.copy(self.grid), -0.1, False, None
        elif stats == error:
            self.grid = oldGrid
            return np.copy(self.grid), -1, False, None

    def reset(self):
        self.last_action = None
        self.grid = np.copy(self.base)
        self.original_indices_row, self.original_indices_col = np.nonzero(self.grid)
        return np.copy(self.grid)

    def render(self, mode='human', close=False):
        if self.last_action != None:
            square = self.last_action%self.nxn
            col = square%self.n
            row = (square-col)//self.n
            val = self.last_action//self.nxn
        for i in range(len(self.grid)):
            for j in range(len(self.grid)):
                if self.last_action != None and i == row and j == col:
                    if val == self.grid[i, j]:
                        print('')
                    else:
                        print('')
                else:
                    print('')
                if j % 3 == 2 and j != len(self.grid)-1:
                    print(' | ')
            if i % 3 == 2 and i != len(self.grid) - 1:
                print('\n-------------------------\n')
            else: 
                print('\n')
        print('\n\n')

        
