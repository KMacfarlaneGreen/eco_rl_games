import numpy as np
import sys
import os


class Two_patch_selection: 
    LEFT = 0
    RIGHT = 1
    STAY = 2
    A = [LEFT,RIGHT, STAY]
    A_DIFF = [-1, 1, 0]
    QUALITY = [5, 2]

    def __init__(self, args, current_path):
        self.num_agents = args['agents_number']  
        self.graph_size = args['graph_size']
        self.state_size = (self.num_agents + len(self.QUALITY))   #should this be size of whole state or fov? Why is there a *2?
        self.agents_positions = []
        self.nodes = []
        self.positions_idx = []  
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

    def reset(self):  # initialize the world

        self.terminal = False
        [self.nodes, self.positions_idx] = self.set_positions_idx()

        # separate the generated position indices
        agents_positions_idx = self.positions_idx

        # map generated position indices to positions
        self.agents_positions = [self.nodes[pos] for pos in agents_positions_idx] #is this different?

        initial_state = list(self.agents_positions + self.QUALITY)  #put quality of patches in initial state

        return initial_state  #this is a list of the initial positions of the agents plus the quality of the patches


    def step(self, agents_actions):
        # update the position of agents
        self.agents_positions = self.update_positions(self.agents_positions, agents_actions)
        
        agent_rewards =[]
        for idx in range(len(self.agents_positions)):
            agent_pos = self.agents_positions[idx]
            ag_reward = self.QUALITY[agent_pos] - self.agents_positions.count(agent_pos)
            agent_rewards.append(ag_reward)

        #reward = sum(agent_rewards)
        reward = agent_rewards
        # check the terminal case - what is my terminal case?

        if sum(reward) == 0:
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

