#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:34:52 2020

@author: epeatfield
"""

import gym
from gym import spaces
import random, copy
from pprint import pprint as pp
import numpy as np
import math
import sys


solved_grid = np.array( [[3,4,1,2,9,7,6,8,5],
              [2,5,6,8,3,4,9,7,1],
              [9,8,7,1,5,6,3,2,4],
              [1,9,2,6,7,5,8,4,3],
              [8,7,5,4,2,3,1,9,6],
              [6,3,4,9,1,8,2,5,7],
              [5,6,3,7,8,9,4,1,2],
              [4,1,9,5,6,2,7,3,8],
              [7,2,8,3,4,1,5,6,9]])

unsolved_grid = np.array([[9,4,1,2,9,7,6,8,5],
                  [2,5,6,8,3,4,9,7,1],
                  [9,8,7,1,5,6,3,2,4],
                  [1,9,2,6,6,5,8,4,3],
                  [8,7,5,4,2,3,1,9,6],
                  [6,3,4,9,1,8,2,5,7],
                  [5,6,3,7,8,9,4,1,2],
                  [4,1,9,5,6,2,2,3,8],
                  [7,2,8,3,4,1,5,6,9]])


def checkIfSolved(grid):
        N = len(grid)
        zeros = np.count_nonzero(grid == 0)
        if not zeros == 0:
            return -1
        #check rows
        for i in range(1, N + 1):
            count = np.count_nonzero(grid == i, axis=1)
            valid = checkOnesInside(count)
            if not valid:
                return 0
            countcols = np.count_nonzero(grid == i, axis = 0)
            validcols = checkOnesInside(countcols)
            if not validcols:
                return 0
    
        #check boxes
        N = len(grid)
        j = int(math.sqrt(N))
        k = N//j
        m = 0
        rowm = m
        rowk = k
        colm = m
        colk = k
        while colk < N+1:
            #for i range(j):
            rowk = k
            rowm = m
            while rowk < N+1:
                box = []
                for col in range(colm, colk):
                    for row in range(rowm, rowk):
                        value = grid[row][col]
                        box.append(value)
                for i in range(9):
                    count = box.count(i+1)
                    if not count == 1:
                        return 0
                rowm += j
                rowk += j
            colk += j
            colm += j
            
        return 1
    
def checkOnesInside(array):
    return all(x == array[0] for x in array)
    
    # @author: https://github.com/artonge/gym-sudoku
def getSolutions(grid, stop=1, col=-1, row=-1, omit=-1):
    N = len(grid)
    isSolved = checkIfSolved(grid)
    if isSolved == 1:
        return grid
        
        #check for empty spaces if no i given in 
    if col == -1:
        for col in range(N):
            for row in range(N):
                if grid[col][row] == 0:
                    break
            if grid[col][row] == 0: 
                break
                
    values = np.arange(1, N+1)
    np.random.shuffle(values)
    solutions = np.empty(shape=(0,N,N))
    for value in values:
        if omit == value:
            continue
        copy = np.copy(grid)
        copy[col][row] = value
        subSol = getSolutions(copy, stop=stop-len(solutions))
        solutions = np.concatenate((solutions, subSol))
        if len(solutions) >= stop:
            return solutions
    return solutions
    
    
class Sudoku_Solve():
    last_action = None
    
    def __init__(self):
        self.last_action = None
        self.grid = []
        self.base = getSolutions(np.zeros(shape=(9,9)))[0]
    
    
            
    def step(self, action):
        self.last_action = action
        last_grid = np.copy(self.grid)
        
        if self.grid[action[0]][action[1]] != 0:
            return np.copy(self.grid), -1, False, None
        
        self.grid[action[0]][action[1]] = action[2]+1
        
        sol = self.checkIfSolved(self.grid)
        
        if sol == -1:
            return np.copy(self.grid), 1, False, None
        if sol == 1:
            return np.copy(self.grid), 1, True, None
        if sol == 0: 
            return np.copy(self.grid), -1, False, None
        
        
        
    def reset(self):
        print("______________Hello")
        self.last_action = None
        self.grid = np.copy(self.base)
        return np.copy(self.grid)
            
print(checkIfSolved(solved_grid))
print(getSolutions(unsolved_grid))
    
