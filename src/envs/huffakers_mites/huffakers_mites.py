import functools
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Discrete, Box

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4
MOVES = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'STAY']
MAX_ITERS = 1000

def env(render_mode = None):
    """env function wraps the environment in wrappers by default."""

    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(render_mode = None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env

class parallel_env(ParallelEnv):
    metadata = {'render.modes':['human'], 'name': 'huffakers_mites_v0'}

    def __init__(self, render_mode = None):
        """
         The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        """
        super().__init__()
        self.pred_agents = 10
        self.prey_agents = 10
        self.dim1 = 4
        self.dim2 = 10
        self.graph_size = self.dim1*self.dim2
        self.state_size = self.graph_size * 3
        self.nodes = [i for i in range(self.graph_size)]  #node list - need mapping from list to grid
        self.quality = np.random.randint(0,10, (self.dim1,self.dim2)) #assign quality to each node (initially random and constant)
        self.render_mode = render_mode
        self.possible_preds = [str(f'predator_{i}') for i in range(self.pred_agents)]
        self.possible_prey = [str(f'prey_{i}') for i in range(self.prey_agents)]
        self.possible_agents = self.possible_preds + self.possible_prey
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(self.pred_agents + self.prey_agents))))

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        #want arrays of pred prey occupancy and quality for agent fov
        return Box(low=np.zeros((3,9)), high = 10*np.ones((3,9)), dtype=np.float32) 
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)    #agents select a node to move to 

    def seed(self, seed = None):
        pass

    def render(self, mode='human'):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        for i in range(self.agent_pop):
            print("Agent {} is at position {} and recieved reward {}".format(i, self.agents_positions[i], self.rewards[self.agents[i]]))

    def close(self):
        pass

    def get_fov(self, agent):
        fov_pred = -1 * np.ones((3,3))   
        fov_prey = -1 * np.ones((3,3)) #two 3x3 observations for predators and prey agents
        fov_quality = -1 * np.ones((3,3))  #start all with -1 padding
        agent_pos = self.agents_positions[agent]

        fov_pred[1][1] = self.map_preds[agent_pos[0]][agent_pos[1]]  #fov centres
        fov_prey[1][1] = self.map_prey[agent_pos[0]][agent_pos[1]]
        fov_quality[1][1] = self.quality[agent_pos[0]][agent_pos[1]]

        #want to pad with -1 if agent is at edge of map
        #currently diagonals also padded as not available for moves - could edit this
        if agent_pos[0] > 0:   #up
            fov_pred[0][1] = self.map_preds[agent_pos[0]-1][agent_pos[1]]
            fov_prey[0][1] = self.map_prey[agent_pos[0]-1][agent_pos[1]]
            fov_quality[0][1] = self.quality[agent_pos[0]-1][agent_pos[1]]
        if agent_pos[0] < self.dim1-1:  #down
            fov_pred[2][1] = self.map_preds[agent_pos[0]+1][agent_pos[1]]
            fov_prey[2][1] = self.map_prey[agent_pos[0]+1][agent_pos[1]]
            fov_quality[2][1] = self.quality[agent_pos[0]+1][agent_pos[1]]
        if agent_pos[1] > 0:    #left
            fov_pred[1][0] = self.map_preds[agent_pos[0]][agent_pos[1]-1]
            fov_prey[1][0] = self.map_prey[agent_pos[0]][agent_pos[1]-1]
            fov_quality[1][0] = self.quality[agent_pos[0]][agent_pos[1]-1]
        if agent_pos[1] < self.dim2-1:  #right
            fov_pred[1][2] = self.map_preds[agent_pos[0]][agent_pos[1]+1]
            fov_prey[1][2] = self.map_prey[agent_pos[0]][agent_pos[1]+1]
            fov_quality[1][2] = self.quality[agent_pos[0]][agent_pos[1]+1]

        return np.hstack((fov_pred, fov_prey, fov_quality))   #how to stack obs arrays?


    def reset(self, seed=None, return_info = False, options= None):
        self.agents = self.possible_agents
        self.agents_positions = {agent: np.hstack((np.random.randint(0, self.dim1), np.random.randint(0,self.dim2))) for agent in self.agents} #positions initialised randomly
        self.map_preds = np.zeros((self.dim1, self.dim2))
        self.map_prey = np.zeros((self.dim1, self.dim2))

        for i in range(self.pred_agents):
            self.map_preds[self.agents_positions[self.possible_preds[i]][0]][self.agents_positions[self.possible_preds[i]][1]] += 1
            
        for i in range(self.prey_agents):
            self.map_prey[self.agents_positions[self.possible_prey[i]][0]][self.agents_positions[self.possible_prey[i]][1]] += 1

        self.agent_fov = {agent: self.get_fov(agent) for agent in self.agents}

        observations = {agent: self.agent_fov[agent] for agent in self.agents}

        self.num_moves = 0

        return observations

    def step(self, actions):
        self.rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        if not actions:
            print('empty')
            self.agemts =[]
            return {}, {}, {}, {}, {}

        #move agents to new positions

        for agent in self.agents:
            if actions[agent] == 0:
                self.agents_positions[agent][0] = max(0, self.agents_positions[agent][0]-1) #up
            elif actions[agent] == 1:
                self.agents_positions[agent][0] = min(self.dim1-1, self.agents_positions[agent][0]+1) #down
            elif actions[agent] == 2:
                self.agents_positions[agent][1] = max(0, self.agents_positions[agent][1]-1)
            elif actions[agent] == 3:
                self.agents_positions[agent][1] = min(self.dim2-1, self.agents_positions[agent][1]+1)
            elif actions[agent] == 4:
                pass
            else:
                raise ValueError("Invalid action")
        
        #update maps
        self.map_preds = np.zeros((self.dim1, self.dim2))
        self.map_prey = np.zeros((self.dim1, self.dim2))

        for i in range(self.pred_agents):
            self.map_preds[self.agents_positions[self.possible_preds[i]][0]][self.agents_positions[self.possible_preds[i]][1]] += 1
            
        for i in range(self.prey_agents):
            self.map_prey[self.agents_positions[self.possible_prey[i]][0]][self.agents_positions[self.possible_prey[i]][1]] += 1

        #calculate updated fov
        self.agent_fov = {agent: self.get_fov(agent) for agent in self.agents}

        #calulate rewards
        for i in range(self.pred_agents):
            pos = self.agents_positions[self.possible_preds[i]]
            self.rewards[self.possible_preds[i]] = self.map_prey[pos[0]][pos[1]]/self.map_preds[pos[0]][pos[1]]

        for i in range(self.prey_agents):
            pos = self.agents_positions[self.possible_prey[i]]
            self.rewards[self.possible_prey[i]] = self.quality[pos[0]][pos[1]]/self.map_prey[pos[0]][pos[1]] - self.map_preds[pos[0]][pos[1]]

        self.num_moves += 1
        env_truncation = self.num_moves >= MAX_ITERS
        truncations = {agent: env_truncation for agent in self.agents}

        if env_truncation:
            self.agents = []

        observations = {agent: self.agent_fov[agent] for agent in self.agents}

        if self.render_mode == 'human':
            self.render()

        return observations, self.rewards, terminations, truncations, infos




            

            