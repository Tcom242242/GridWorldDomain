#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random
import numpy as np
import pdb

_stay = 0
_right = 1
_left = 2
_up = 3
_down = 4

class Agent:
    def __init__(self, e=0.1, a=0.1, row=3, col=3):
        self.e, self.a = e, a
        self.x, self.y, self.last_x, self.last_y= 0, 0, 0, 0
        self.world_col ,self.world_row = col, row
        self.last_action = 0
        self.average_reward = 0.0
        self.average_reward_list = []
        self.action_selection_method = "egreedy"
        self.qtable = []
        self._init_qtable(col, row)

    def _init_qtable(self,col, row):
        '''
            each section has 5 actions
            0 : stay
            1 : right
            2 : left
            3 : up
            4 : down
        '''
        self.qtable = [[[0.0 for i in range(5)] for k in range(row)] for j in range(col)]

        for x in range(row):
            for y in range(col):
                if y == 0: self.qtable[x][y][_up] = -1.0
                if y == col-1: self.qtable[x][y][_down] = -1.0
                if x == 0: self.qtable[x][y][_left] = -1.0
                if x == row-1: self.qtable[x][y][_right] = -1.0


    def get_pos(self):
        return np.array([self.x, self.y])
        
    def _update_q_table(self, reward):
        self.qtable[self.last_x][self.last_y][self.last_action] = self.a * reward + (1.0 - self.a)*self.qtable[self.last_x][self.last_y][self.last_action]

    def select_action(self):
        action = None
        if self.action_selection_method == "egreedy":
            action = self._e_greedy()

        self.last_action = action 
        return action

    def do_action(self):
        x, y = self.x, self.y
        if self.last_action == _up:
            y -= 1
        elif self.last_action == _down:
            y += 1
        elif self.last_action == _right:
            x += 1 
        elif self.last_action == _left:
            x -= 1

        self.last_x , self.last_y = self.x, self.y
        self.x, self.y = x, y

    def _e_greedy(self):
        if random.random() <= self.e:
            action = self.random_select_action()
        else:
            action = self.greedy_select_action()

        return action

    def get_reward(self, reward, cycle):
        self.calc_average_reward(reward, cycle)
        self._update_q_table(reward)
   

    def calc_average_reward(self, reward, cycle):
        self.average_reward = (cycle / (cycle + 1.0)) * self.average_reward + (1.0 / (cycle + 1.0)) * reward
        self.average_reward_list.append(self.average_reward)

    def greedy_select_action(self):
        max_action_reward = -1.0
        max_action = 0

        for i, q in enumerate(self.qtable[self.x][self.y]):
            if (q > max_action_reward):
                max_action_reward = q
                max_action = i

        return max_action 

    def random_select_action(self):
        action = random.randint(0,len(self.qtable[self.x][self.y])-1)
        while self.qtable[self.x][self.y][action] == -1:
            action = random.randint(0,len(self.qtable[self.x][self.y])-1)
                
        return action


