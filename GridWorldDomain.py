#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random
import numpy as np
import pdb
from Agent import Agent
from tqdm import tqdm
import time
import json

"""
    GridWorldDomain 
    
"""

class GridWorldDomain:
    agents = []
    def __init__(self):
        self._init_world()
        self._init_agents()
        self.max_cycle = 1000
        self.poi_value = 10 

    def _init_world(self):
        self.col = 5  # grid's col
        self.row = 5  # grid's col
        self.poi_pos_list = np.array([5, 5])    

    def _init_agents(self):
        """ initialize agents
            all agent select action by epsilon greedy .
            e : epsilon 
            a : alpha (learning rate)
        """
        self.agents = [Agent(e=0.1, a=0.1, row=self.row, col=self.col) for i in range(2)] 
    
    def calc_dist_to_poi(self,agent):
        """ return minimum dist between position of a agent and a poi
        
        """
        mini_dist = 100000  
        for poi in self.poi_pos_list:
            mini_dist = np.linalg.norm(agent.get_pos() - poi)

        return mini_dist

    def calc_reward(self, agent):
        dist = self.calc_dist_to_poi(agent)
        if dist < 2:
            return self.poi_value
        elif dist < 10:
            return self.poi_value/dist
        else:
            return 0

    def run(self):
        for cycle in tqdm(range(self.max_cycle)):
            actions = [agent.select_action() for agent in self.agents]   # each agent acts
            for i, agent in enumerate(self.agents):  # each agent gets reward
                agent.do_action()
                agent.get_reward(self.calc_reward(agent), cycle)
                # print(agent.average_reward)
                # print(agent.get_pos())
        self.save_log()
        
    
    def save_log(self):
        """ Save log like rewards and so on
           
        """ 
        result = {}
        for i, agent in enumerate(self.agents):
            result["agent"+str(i)+"ave_reward_list"] = agent.average_reward_list

        f = open("agent_ave_reward.json", "w")
        json.dump(result, f)

if __name__ == '__main__':
    game = GridWorldDomain()
    game.run()

