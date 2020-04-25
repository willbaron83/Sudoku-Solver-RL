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

resolved = 0
unfinished = 1
error = 2

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
    #check rows
    for i in range(1, N + 1):
        count = np.count_nonzero(grid == i, axis=1)
        valid = checkOnesInside(count)
        if not valid:
            return False
        countcols = np.count_nonzero(grid == i, axis = 0)
        validcols = checkOnesInside(countcols)
        if not validcols:
            return False

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
                    return False
            rowm += j
            rowk += j
        colk += j
        colm += j
        
    return True

def checkOnesInside(array):
    return all(x == array[0] for x in array)


# def getSolutions(grid):
#     isSolved = checkIfSolved(grid)
#     if isSolved:
#         return grid
        
    
        
print(checkIfSolved(solved_grid))
print(checkIfSolved(unsolved_grid))
#def checkInVector(self, grid):
    
