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

from torch.multiprocessing import Queue, Lock

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from agent import Agent
from mind import Mind

class Environment:
    def __init__(self, size, num_actions = 2, name = None, 
            max_iteration = 5000, num_agents = 20, lock = None):
       
        node_num = self.size = size
        self.graph = nx.cycle_graph(node_num)
        self.nodes = list(self.graph.nodes)
        self.num_agents = num_agents
        input_size = 3     #this should be the size of the observation space - how should the observation space be represented [num_left, num_loc, num_right]?
        if lock:
            self.lock = lock
        else:
            self.lock = Lock()
        self.mind = Mind(input_size, num_actions, self.lock, Queue())

        #to do: sort mind function:
        #- inputs, outputs, handling each agent's individual mind
        weights = self.mind.network.state_dict()  #needed? what does this do?

        self.max_iteration = max_iteration
        self.crystal = np.zeros((max_iteration, node_num, 1)) 
        self.history = []        #think about what quantities I want to save
        #self.id_track = []       #how to define, intitialise and store them 
        self.records = []
        self.q_values = np.zeros((max_iteration, num_agents, 2))

        if name:
            self.name = name

        if not os.path.isdir(str(self.name)):
            os.mkdir(str(self.name))
            os.mkdir(str(self.name)+'/episodes')
            self.map, self.agents, self.id_to_agent = self._generate_map()  #deleted self.loc_to_agent
            self._set_initial_states()
            self.crystal = np.zeros((max_iteration, node_num, 1)) #in segregation crystal is an array which saves the agent type, id and age at each location at each iteration
                                                                  #for me I just want to save the number of agents at each node at each iteration (map)
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
                
        self.map[int(i)] -= 1         #remove an agent from the previous node
        self.map[int(i_n)] += 1       #add this agent to the new node 
        agent.set_loc(to)         #set this new node location to the agent
        #self.loc_to_agent[to] = agent        
        #del self.loc_to_agent[loc]   #delete the previous entry in the dictionary - maybe this is causing problems if multiple agents are at the same location?
        return self.map.copy()       #correct?

    def step(self, agent, act):
        #mark as updated
        #function for calculating the local reward of each agent cumulative and global entropy reward should be calculated somewhere else - mind?
        (i) = agent.get_loc() # current location - change to graph
        #print('loc', i)
        #print('loc to agent length', len(self.loc_to_agent))
        #assert self.loc_to_agent[(i)]
        (di) = act     #action 
        (i_n) = self._add((i), (di))    #gives new location
        agent.set_decision((i_n))
        self.move(agent)
        
        if self.map[int(i_n)] == 1.0:   #if agent has moved to a node where it is the only one present it receives a positive reward
            rew = 1                     
            assert rew != None                             
        elif self.map[int(i_n)] > 1.0:  #if agent moves to a node where there are also other agents then it recieves a negatibve reward
            rew = -1
            assert rew != None
        else:
            assert rew != None     #raise assertion error if node value at agent location is zero
        done = False               #done set here
        self.update_agent(agent, rew, done)
        agent.clear_decision()
        return rew


    def update_agent(self, agent, rew, done):   
        state = self.get_agent_state(agent)  #called after agent has moved
        agent.set_next_state(state)     #therefore set to next state             
        agent.update(rew, done)   #pushes transition to memory
        return rew

    def update(self):
        #mark as updated
        self.iteration += 1
        self.history.append(self.map.copy())     #do we want our history to represent the whole graph?
        cumulative_reward = self.records[self.iteration-1]["tot_reward"]
        self.mind.train()   #does this need an alternative input to vals_to_names? 
        #need to work out how training functions will work with each individual being trained in a decentralised manner
        agent_ids = []
        agent_locs = []
        for agent in self.agents:
            idx = agent.get_id()
            i = agent.get_loc() 
            self.crystal[self.iteration - 1, int(i)] += 1 #add 1 to the entry at the agent location to give the number of agents located on the node
            agent_ids.append(str(idx))
            agent_locs.append(str(i))

        #self.id_track.append(id_track)
        agent_ids = " ".join(agent_ids)
        agent_locs = " ".join(agent_locs)
        iteration_reward = str(cumulative_reward)

        with open("%s/episodes/agent_trajectory.csv" % self.name, "a") as f:
            f.write("%s, %s, %s, %s\n" % (self.iteration, agent_ids, agent_locs, iteration_reward)) #maybe not the nicest way to save these quantities so could try to improve

        if self.iteration == self.max_iteration - 1:
            losses = self.mind.get_losses()
            np.save("%s/episodes/loss.npy" % self.name, np.array(losses))     #Will eventually want to save the training losses

    #def shuffle(self):
        # do I need this? - not accessed
        #what purpose does this serve different to generate map?
        #map = np.zeros(self.size)
        #loc_to_agent = {}    #this is where loc_to_agent comes in

        #locs = [(i) for i in range(self.map.shape[0]) if self.map[i] == 0]   
        #what locations is this assigning and why? 
        #random.shuffle(locs)
        #id_track = np.zeros(self.map.shape)
        #for i, agent in enumerate(self.agents):
            #loc = locs[i]
            #loc_to_agent[loc] = agent
            #id_track[loc] = agent.get_id()


        #self.map = map
         #self.loc_to_agent =  loc_to_agent
        #self._set_initial_states()
        #self.history = [map.copy()]
        #self.id_track = [id_track]
        #self.records = []
        #self.iteration = 0

    def record(self, rews):
        self.records.append(rews)    #rewards goes to records

    def save(self, episode):    
        f = gzip.GzipFile('%s/crystal.npy.gz' % self.name, "w")
        np.save(f, self.crystal)
        f.close()

    def save_qs(self, episode):    
        f = gzip.GzipFile('%s/q_values.npy.gz' % self.name, "w")
        np.save(f, self.q_values)
        f.close()


    def save_agents(self):
        self.lock.acquire()    #what is lock? Need to edit how things are saved because what I want to save is different 
        pickle.dump(self.agents, open("agents/agent_%s.p" % (self.name), "wb" ))
        self.lock.release()

    def get_agent_state(self, agent):
        #think this returns the state of the square observation or 'field of view' for each agent
        #in that case I need to change it to return the current node location and number of agents at that location
        #mark as updated
        #extend this to include number of agents at neighbouring nodes - return [num_left, num_loc, num_right]
        (i)= agent.get_loc()
        if i == 0.0:
            left_loc = 99.0
        else:
            left_loc = i - 1.0 
        if i < 99.0:
            right_loc = i + 1.0
        else:
            right_loc = 0.0
        n_left = self.map[int(left_loc)].copy()
        n_right = self.map[int(right_loc)].copy()
        #print('loc',i)
        n_loc = self.map[int(i)].copy()
        fov = [n_left, n_loc, n_right]
        #print('fov',fov)
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
        # loc_to_agent = location of each agent   (currently removed - not sure needed for me) 
        # id_to_agent = id of each agent
        # work out how to include mind networks
        # Mark as updated
        map = np.zeros(self.size)   #index corresponds to graph node i.e. map[0] = 1 means there is one agent on node zero
        #loc_to_agent = {}
        id_to_agent = {}
        agents = []
        idx = 0
        init_nodes = np.zeros(self.num_agents) #initialise agent locations (gives initial node loaction for each agent i.e. init_node[0] = 10 means agent 0 begins on node 10) 
        for j in range(0, self.num_agents):   #needs to be outside loop over nodes so doesn't change each time
            init_nodes[j] = np.random.choice(self.nodes)
            mind = self.mind
            agent = Agent(idx, (init_nodes[j]), mind)
            #loc_to_agent[(init_nodes[j])] = agent   #float loc
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
        return map, agents, id_to_agent     #deleted loc_to_agent

    def _add(self, loc, act):
        #mark completed
        loc
        act  #action (0=left, 1=right)
        if act == 0:
            #move left
            if loc == 0.0:
                to = 99.0
            else:
                to = loc - 1.0   
        if act == 1:
            #move right
            if loc < 99.0:
                to = loc + 1.0
            else:
                to = 0.0
        return to #(i_n)

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