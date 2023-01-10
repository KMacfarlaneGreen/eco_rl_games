import random
import operator
import numpy as np
#import pygame    #need this or just for displays?
import sys
import os


class Ringantisocial:
    #is this a better way to define agent movements and actions? 
    LEFT = 0
    RIGHT = 1
    #STAY = 2
    A = [LEFT, RIGHT]
    A_DIFF = [-1, 1]

    def __init__(self, args, current_path):
        #self.game_mode = args['game_mode']     #keep and adapt these for flexible running parameters 
        #self.reward_mode = args['reward_mode']
        self.num_agents = args['agents_number']  #look back at the landmarks code when thinking about resource availability
        self.graph_size = args['graph_size']
        self.state_size = (self.num_agents) #* 2   #should this be size of whole state or fov? Why is there a *2?
        self.agents_positions = []

          #do I need this render at all - could save images while running instead of plotting them after? Which is better?
           #is this how things are saved?
        # enables visualizer - think to begin with I probably want to get rid of all the visualisation functions

        self.nodes = []
        self.positions_idx = []  #do these link to the agents through order?

        # self.agents_collide_flag = args['collide_flag']
        # self.penalty_per_collision = args['penalty_collision']
        self.num_episodes = 0
        self.terminal = False

    def set_positions_idx(self):
        #updated
        nodes = [i for i in range(0, self.graph_size)]

        positions_idx = []

        #if self.game_mode == 0:   #not sure what different game and reward modes are?
            # , e.g.,
            # positions_idx = [0, 6, 23, 24] where 0 6 23 24 are positions
            # of agents
            #positions_idx = []

        positions_idx = np.random.choice(len(nodes), size = self.num_agents,
                                             replace=True)

        return [nodes, positions_idx]  #returns array of node locations and the positions of the agents on the graph
        #this could be a list of nodes and the nodes where agents are located (equivalent to map - want to save)
        #still need to calculate the agents field of view to go into the NN or assume it can view the whole graph 

    def reset(self):  # initialize the world

        self.terminal = False
        [self.nodes, self.positions_idx] = self.set_positions_idx()

        # separate the generated position indices
        agents_positions_idx = self.positions_idx

        # map generated position indices to positions
        self.agents_positions = [self.nodes[pos] for pos in agents_positions_idx] #is this different?

        initial_state = list(self.agents_positions)  #why would you sum the positions?

        return initial_state  #this is a list of the initial positions of the agents
        #is this what we want or should it return a map of the occupancy of the nodes (start with positions and move to occupancy)


    def step(self, agents_actions):
        # update the position of agents
        #start with team reward and change to individual reward 
        self.agents_positions = self.update_positions(self.agents_positions, agents_actions)

        if len(set(self.agents_positions)) == len(self.agents_positions):
            reward = 1
        
        else:
            reward = -1 

        # check the terminal case - what is my terminal case?
        if reward == 0:
            self.terminal = True
        else:
            self.terminal = False


        new_state = list(self.agents_positions)

        return [new_state, reward, self.terminal]

    def update_positions(self, pos_list, act_list):
        positions_action_applied = []
        for idx in range(len(pos_list)):  #for each agent
            pos_act_applied = pos_list[idx] + self.A_DIFF[act_list[idx]]
            # checks to make sure the new pos in inside the grid
            if pos_act_applied < 0:    #change to be ring boundary conditions
                pos_act_applied = self.graph_size
            if pos_act_applied > self.graph_size:
                pos_act_applied = 0
            positions_action_applied.append(pos_act_applied)

        final_positions = positions_action_applied

        return final_positions

    def action_space(self):
        return len(self.A)


    def find_frequency(self, a, items):
        freq = 0
        for item in items:
            if item == a:
                freq += 1

        return freq