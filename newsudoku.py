#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:51:26 2020

@author: epeatfield
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:30:15 2020

@author: epeatfield
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 21:09:31 2020

@author: https://github.com/artonge/gym-sudoku
"""

import gym
from gym import spaces
import math
import random, copy
from pprint import pprint as pp
import numpy as np
import sys
# from sudoku import Sudoku

resolved = 0
unfinished = 1
error = 2

def checkSolution(grid):
	N = len(grid)
	# zeros = np.count_nonzero(grid==0)
	# if not zeros == 0:
	# 	return unfinished	
	#check rows
	# for i in range(1, N+1):
	# 	count = np.count_nonzero(grid == i, axis = 1)
	# 	valid = checkOnesInside(count)
	# 	if not valid: 
	# 		return error
	# 	countcols = np.count_nonzero(grid == i, axis = 0)
	# 	validcols = checkOnesInside(countcols)
	# 	if not validcols:
	# 		return error
	for i in range(N):
		for j in range(N):
			# If a case is not filled, the sudoku is not finished
			if grid[i][j] == 0:
				return unfinished
			n = N//3
			iOffset = i//n*n
			jOffset = j//n*n
			square = grid[ iOffset:iOffset + n , jOffset:jOffset + n].flatten()
			# Check uniqueness
			uniqueInRow    = countItem(grid[i], grid[i, j])  == 1
			uniqueInCol    = countItem(grid[:,j:j+1].flatten(), grid[i, j]) == 1
			uniqueInSquare = countItem(square, grid[i, j]) == 1

			if not (uniqueInRow and uniqueInCol and uniqueInSquare):
				return error
			
		
	return resolved

def checkOnesInside(array):
	return all(x <= 1 for x in array)


# Count the number of time the item appears in a vector
def countItem(vector, item):
	count = 0
	for item2 in vector:
		if item2 == item: count += 1
	return count


# Recursivly find all solutions (backtracking)
# @param stopAt make the backtracking stop when it found x solutions
# @param i, j force to start the backtracking from the case (i, j)
# @param omit prevent looking into a possibility
def getSolutions(grid, stopAt=1, i=-1, j=-1, omit=-1):
	N = len(grid)
	check = checkSolution(grid)
	# Check if grid is resolve or if there is an error
	if check == resolved:
		return np.array([grid], dtype=int)
	if check == error:
		return np.empty(shape=(0,N,N), dtype=int)

	# If i and j are not setted, get the first empty spot and start backtracking from it
	if i == -1:
		for i in range(N):
			for j in range(N):
				# If not empty spot continue
				if grid[i, j] == 0: break
			if grid[i, j] == 0: break

	# Randomize possible values
	values = np.arange(1, N+1)
	np.random.shuffle(values)
	# Try all possiblities from those values until we reach the max nb of solutions asked by stopAt
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
	metadata = {'render.modes': ['human']}
	last_action = None

	# Make a random grid and store it in self.base
	def __init__(self):
		# The box space is continuous. This don't apply to a sudoku grid, but there is no other choices
		self.observation_space = spaces.Box(1,9, shape=(9,9))
		self.action_space = spaces.Tuple((spaces.Discrete(9), spaces.Discrete(9), spaces.Discrete(9)))
		# Get a random solution for an empty grid
		self.grid = []
		self.original_indices_row = []
		self.original_indices_col = []
		self.base = getSolutions(np.zeros(shape=(9,9)))[0]
		# Get all positions in random order, to randomly parse the grid
		N = len(self.base)
		positions = []
		for i in range(N):
			for j in range(N):
				positions.append((i, j))
		np.random.shuffle(positions)

		count = 0
		# Try to put 0 instead of the original value for all positions
		# Stop after 40 --> medium difficulty
		# This is slow after 40 because, the algorithm looks for 1 solutions when there is none,
		# so it realy check all the possibilities...
		for i, j in positions:
			if count > 3:
				break
			oldValue = self.base[i, j]
			self.base[i, j] = 0
			solutions = getSolutions(self.base, stopAt=2, i=i, j=j, omit=oldValue)
			if len(solutions) == 0:
				count += 1
			else:
				# if more than one solution undo
				self.base[i, j] = oldValue
		##Store i, j values for original grid
		


	# @return
	# 	- a copy of the grid to prevent alteration from the user
	# 	- a reward: - negative if action leads to an error
	#	            - positive if action is correct or grid is resolved
	def step(self, action):
		if self.last_action != None and self.last_action == action: 
			return np.copy(self.grid), -0.5, False, None

		self.last_action = action
		oldGrid = np.copy(self.grid)

		# The user can't replace a value that was already set originally
		for i in range(len(self.original_indices_row)):
			if action[0] == self.original_indices_row[i] and action[1] == self.original_indices_col[i]:
				return np.copy(self.grid), -1, False, None
		 	# elif self.grid[action[0], action[1]] != 0: 
			# print("Replacing old value")

		# We add one to the action because the action space is from 0-8 and we want a value in 1-9
		if self.grid[action[0], action[1]] == action[2]+1:
			return np.copy(self.grid), -1, False, None

		self.grid[action[0], action[1]] = action[2]+1

		stats = checkSolution(self.grid)
		# If grid is complet or correct, return positive reward
		if stats == resolved:
			return np.copy(self.grid), 1, True, None
		elif stats == unfinished:
			return np.copy(self.grid), -0.05, False, None
		if stats == error:
			# If move is wrong, return to old state, and return negative reward
			self.grid = oldGrid
			return np.copy(self.grid), -1, False, None


	# Replace self.grid with self.base
	# Creating a new grid at every reste would be expensive
	def reset(self):
		self.last_action = None
		self.grid = np.copy(self.base)
		self.original_indices_row, self.original_indices_col = np.nonzero(self.grid)
		# print(self.original_indices_row, self.original_indices_col, self.grid)
		return np.copy(self.grid)


	def render(self, mode='human', close=False):

		for i in range(len(self.grid)):
			for j in range(len(self.grid)):
				if self.last_action != None and i == self.last_action[0] and j == self.last_action[1]:
					if self.last_action[2] == self.grid[i, j]:
						sys.stdout.write('\033[92m' + str(self.last_action[2]) + '\033[0m')
					else:
						sys.stdout.write('\033[91m' + str(self.last_action[2]) + '\033[0m')
				else:
					sys.stdout.write(str(self.grid[i, j]))
				if j % 3 == 2 and j != len(self.grid)-1:
					sys.stdout.write(' | ')
			if i % 3 == 2 and i != len(self.grid)-1:
				sys.stdout.write('\n---------------\n')
			else:
				sys.stdout.write('\n')
		sys.stdout.write('\n\n')
		sys.stdout.flush()


# env = SudokuEnv()
# env.reset()

# print(env.grid)

# grid = np.array(
# [[0,0,0,4,0,9,0,0,1],
# [0,0,4,0,3,0,0,2,0],
# [0,7,2,0,5,1,0,0,6],
# [4,2,1,0,0,5,6,0,0],
# [8,0,0,0,0,2,0,0,0],
# [3,0,0,9,0,0,0,0,0],
# [0,1,0,5,7,4,0,0,0],
# [5,0,6,0,0,3,0,0,7],
# [0,0,3,0,9,0,0,1,0]])
# #
# print(getSolutions(env.grid))
