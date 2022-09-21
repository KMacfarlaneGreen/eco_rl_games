import os
import gzip
import math
import copy
import pickle
import random
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

from multiprocessing import Queue, Lock

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from agent import Agent
from mind import Mind

class Environment:
    def __init__(self, size, num_actions = 2, name = None, 
            max_iteration = 5000, num_agents = 20):
       
        node_num = self.size = size
        self.graph = nx.cycle_graph(node_num)
        self.nodes = list(self.graph.nodes)
        self.num_agents = num_agents
        input_size = 1
        self.mind = Mind(input_size, num_actions)

        #to do: sort mind function:
        #- inputs, outputs, handling each agent's individual mind

        #self.A_mind = Mind(input_size, num_actions, self.lock, Queue())
        #self.B_mind = Mind(input_size, num_actions, self.lock, Queue())
        #for segregation the agent_range defines the window that each agent can observe which is then input to that nn
        self.max_iteration = max_iteration

        #self.vals = [-1, 0, 1, 2]  #possible values for grid squares in segregation case
        #self.names_to_vals = {"void": -2, "A": -1, "free": 0, "B": 1, "prey": 2}
        #self.vals_to_names = {v: k for k, v in self.names_to_vals.items()}
        #self.vals_to_index = {-1: 0, 0: 1, 1: 2, 2: 3}

        self.crystal = np.zeros((max_iteration, node_num, 1))  #id  -should there be other info in this?
        #what actually is this crystal object? Is this what's being saved?

        self.history = []        #think about what quantities I want to save
        self.id_track = []       #how to define, intitialise and store them 
        self.records = []

        if name:
            self.name = name

        if not os.path.isdir(str(self.name)):
            #os.mkdir(str(self.name))
            #os.mkdir(str(self.name)+'/episodes')
            self.map, self.agents, self.loc_to_agent, self.id_to_agent = self._generate_map()
            self._set_initial_states()
            #self.mask = self._get_mask()
            self.crystal = np.zeros((max_iteration, nodes, 2)) # tr,  id
            self.iteration = 0
        else:
            assert False, "There exists an experiment with this name."

    def get_agents(self):
        return self.agents

    def get_map(self):
        return self.map.copy()

    def move(self, agent):
    #mark as updated
    #moves an agent to new node location following a decision to move left or right
        (i) = loc = agent.get_loc()     #only need single index to correspond to single node 
        (i_n) = to = agent.get_decision()  #decision returns the new location after performing action 0 for move left and 1 for move right
                
        self.map[i] -= 1         #remove an agent from the previous node
        self.map[i_n] += 1       #add this agent to the new node 
        agent.set_loc(to)         #set this new node location to the agent
        self.loc_to_agent[to] = agent        
        del self.loc_to_agent[loc]   #delete the previous entry in the dictionary

    def step(self, agent, act):
         #function for calculating the local reward of each agent cumulative and global entropy reward should be calculated somewhere else - mind?
        (i) = agent.get_loc() # current location - change to graph
        assert self.loc_to_agent[(i)]
        (di) = act     #action 
        (i_n) = self._add((i), (di))
        agent.set_decision((i_n))
        self.move(agent)
        
        if self.map[i_n] == 1.0:   #if agent has moved to a node where it is the only one present it receives a positive reward
            rew = 1                     
            assert rew != None                             
        elif self.map[i_n] > 1.0:  #if agent moves to a node where there are also other agents then it recieves a negatibve reward
            rew = -1
            assert rew != None
        else:
            assert rew != None     #raise assertion error if node value at agent location is zero
        done = False
        self.update_agent(agent, rew, done)
        agent.clear_decision()
        return rew


    def update_agent(self, agent, rew, done):    #what are the different functions of step, update_agent and update?
        state = self.get_agent_state(agent)
        agent.set_next_state(state)                 #is this set_next_state in mind?
        name = self.vals_to_names[agent.get_type()]   #? - don't have different types of agents
        agent.update(rew, done)
        return rew

    def update(self):
        #to do: think about mind and how that will work, inputs, outputs, what to save
        self.iteration += 1
        self.history.append(self.map.copy())     #do we want our history to represent the whole graph?

        self.A_mind.train(self.names_to_vals["A"])    #need to work out how training functions will work with each individual being trained in a decentralised manner
        self.B_mind.train(self.names_to_vals["B"])    #lots of the rest of this function we won't need
        a_ages = []
        a_ids = []
        b_ages = []
        b_ids = []
        id_track = np.zeros(self.map.shape)
        self.deads = []
        for agent in self.agents:
            typ = agent.get_type()
            age = agent.get_age()
            idx = agent.get_id()

            if agent.is_alive():
                i, j = agent.get_loc()
                tr = agent.get_time_remaining()
                id_track[i, j] = idx
                self.crystal[self.iteration - 1, i, j] = [typ, age, tr, idx]
            else:
                self.deads.append([agent.get_type(), agent.get_id()])

            type = agent.get_type()
            if type == self.names_to_vals["A"]:
                a_ages.append(str(age))
                a_ids.append(str(idx))
            else:
                b_ages.append(str(age))
                b_ids.append(str(idx))

        self.id_track.append(id_track)
        a_ages = " ".join(a_ages)
        b_ages = " ".join(b_ages)

        a_ids = " ".join(a_ids)
        b_ids = " ".join(b_ids)

        with open("%s/episodes/a_age.csv" % self.name, "a") as f:
            f.write("%s, %s, %s\n" % (self.iteration, a_ages, a_ids))

        with open("%s/episodes/b_age.csv" % self.name, "a") as f:
            f.write("%s, %s, %s\n" % (self.iteration, b_ages, b_ids))

        if self.iteration == self.max_iteration - 1:
            A_losses = self.A_mind.get_losses()
            B_losses = self.B_mind.get_losses()
            np.save("%s/episodes/a_loss.npy" % self.name, np.array(A_losses))
            np.save("%s/episodes/b_loss.npy" % self.name, np.array(B_losses))

    def shuffle(self):
        #think the shuffles the order of agents for iterations? - do I need this?
        #what purpose does this serve different to generate map?
        map = np.zeros(self.size)
        loc_to_agent = {}    #this is where loc_to_agent comes in

        locs = [(i, j) for i in range(self.map.shape[0]) for j in range(self.map.shape[1]) if self.map[i, j] == 0]
        random.shuffle(locs)
        id_track = np.zeros(self.map.shape)
        for i, agent in enumerate(self.agents):
            loc = locs[i]
            loc_to_agent[loc] = agent
            id_track[loc] = agent.get_id()


        self.map, self.loc_to_agent = map, loc_to_agent
        self._set_initial_states()
        self.history = [map.copy()]
        self.id_track = [id_track]
        self.records = []
        self.iteration = 0

    def record(self, rews):
        self.records.append(rews)    #rewards goes to records

    def save(self, episode):     #work out how and what to save
        f = gzip.GzipFile('%s/crystal.npy.gz' % self.name, "w")
        np.save(f, self.crystal)
        f.close()

    def save_agents(self):
        self.lock.acquire()    #what is lock? Need to edit how things are saved because what I want to save is different 
        pickle.dump(self.agents, open("agents/agent_%s.p" % (self.name), "wb" ))
        self.lock.release()

    def get_agent_state(self, agent):
        #think this returns the state of the square observation or 'field of view' for each agent
        #in that case I need to change it to return the current node location and number of agents at that location
        hzn = self.hzn
        i, j = agent.get_loc()
        fov = np.zeros((2 * hzn + 1, 2 *  hzn + 1)) - 2
        if self.boundary_exists:
            start_i, end_i, start_j, end_j = 0, 2 * hzn + 1, 0, 2 * hzn + 1
            if i < hzn:
                start_i = hzn - i
            elif i + hzn - self.size[0] + 1 > 0:
                end_i = (2 * hzn + 1) - (i + hzn - self.size[0] + 1)
            if j < hzn:
                start_j = hzn - j
            elif j + hzn - self.size[1] + 1 > 0:
                end_j = (2 * hzn + 1) - (j + hzn - self.size[1] + 1)
            i_upper = min(i + hzn + 1, self.size[0])
            i_lower = max(i - hzn, 0)

            j_upper = min(j + hzn + 1, self.size[1])
            j_lower = max(j - hzn, 0)

            fov[start_i: end_i, start_j: end_j] = self.map[i_lower: i_upper, j_lower: j_upper].copy()
        else:
            for di in range(-hzn, hzn+1):
                for dj in range(-hzn, hzn+1):
                    fov[hzn + di, hzn + dj] = self.map[(i+di) % self.size[0], (j+dj) % self.size[1]]

        fov[hzn, hzn] = agent.get_type()
        return fov

    def _to_csv(self, episode):
        #work out what want to save and how and then adapt this 
        with open("episodes/%s_%s.csv" % (episode, self.name), 'w') as f:
            f.write(', '.join(self.records[0].keys()) + '\n')
            proto = ", ".join(['%.3f' for _ in range(len(self.records[0]))]) + '\n'
            for rec in self.records:
                f.write(proto % tuple(rec.values()))



    def _generate_map(self):
        #generates representation of environment in a given state
        #rewrite this function to give graph with a given number of agents on each node
        #where:
        # map = graph with n agents on each node (state of environment)
        # agents = agents
        # loc_to_agent = location of each agent   what is the structure of these dicionaries? 
        # id_to_agent = id of each agent
        # work out how to include mind networks
        # Mark as updated
        map = np.zeros(self.size)   #index corresponds to graph node i.e. map[0] = 1 means there is one agent on node zero
        loc_to_agent = {}
        id_to_agent = {}
        agents = []
        idx = 0
        init_nodes = np.zeros(self.num_agents) #initialise agent locations (gives initial node loaction for each agent i.e. init_node[0] = 10 means agent 0 begins on node 10) 
        for j in range(0, self.num_agents):   #needs to be outside loop over nodes so doesn't change each time
            init_nodes[j] = np.random.choice(self.nodes)
            mind = self.mind
            agent = Agent(idx, (init_nodes[j]), mind)
            loc_to_agent[(init_nodes[j])] = agent
            id_to_agent[idx] = agent
            agents.append(agent)
            idx += 1

        for i in enumerate(map):
            for x in range(0, self.num_agents):
                if i[0] == init_nodes[x]:
                    map[i[0]] += 1
            
            #val = np.random.choice(self.vals, p=self.probs)   #val = number of agents on node i
                                                                #not a random choice - should this come after or within agent loop/assignation
                                                                #similar structure if val not == 0?
                                                                #how to assign number of agents to graph originally?
        return map, agents, loc_to_agent, id_to_agent

    def _add(self, loc, act):
        #mark completed
        i = loc
        di = act  #action (0=left, 1=right)
        if di == 0:
            #move left
            if loc == 0.0:
                (i_n) = to = 99.0
            else:
                (i_n) = to = loc - 1   
        if di == 1:
            #move right
            if loc < 99.0:
                (i_n) = to = loc + 1
            else:
                (i_n)= to = 0.0
        return (i_n)

    def predefined_initialization(self, file):
        # not sure what this does
        with open(file) as f:
            for i, line in enumerate(f):
                if not i:
                    keys = [key.strip() for key in line.rstrip().split(',')]
                line.rstrip().split(',')

    def _set_initial_states(self):
        for agent in self.agents:
            state = self.get_agent_state(agent)
            agent.set_current_state(state)

    def _count(self, arr):

        # not sure what this does - don't think I need it 

        cnt = np.zeros(len(self.vals))
        arr = arr.reshape(-1)
        for elem in arr:
            if elem in self.vals_to_index:
                cnt[self.vals_to_index[elem]] += 1
        return cnt