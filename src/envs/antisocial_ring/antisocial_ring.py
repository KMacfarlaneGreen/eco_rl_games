import functools
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Discrete, Box

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

LEFT = 0
RIGHT = 1
STAY = 2
MOVES = ['LEFT', 'RIGHT', 'STAY']
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
    metadata = {'render.modes': ['human'], 'name': "antisocial_ring_v0", 'observability': ['full', 'partial']}

    def __init__(self, render_mode = None, observability = 'full'):
        """
         The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        """
        super().__init__()
        self.agent_pop = 5
        self.graph_size = 20
        self.state_size = self.graph_size   
        self.nodes = [i for i in range(self.graph_size)]
        self.render_mode = render_mode
        self.observability = observability
        self.possible_agents = [str(i) for i in range(self.agent_pop)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(self.agent_pop))))

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if self.observability == 'full':
            return Box(low=np.zeros((1+self.graph_size)), high = self.graph_size*np.ones((1+self.graph_size)), dtype=np.float32) #Dict(Dict({Discrete(self.graph_size), Box(low=np.zeros((self.graph_size)), high = self.agent_pop*np.ones((self.graph_size)), dtype=np.float32)})) 
        elif self.observability == 'partial':
            return Box(low=np.zeros((4)), high = self.graph_size*np.ones((4)), dtype=np.float32) #Dict(Dict({Discrete(self.graph_size), Box(low=np.zeros((3)), high = self.agent_pop*np.ones((3)), dtype=np.float32)}))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)

    def seed(self, seed=None):
        pass

    def render(self, mode='human'):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        for i in range(self.agent_pop):
            print("Agent {} is at position {} and recieved reward {}".format(i, self.agents_positions[i], self.rewards[self.agents[i]]))  #check this later 

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents[:]
        self.agents_positions = {agent: np.random.choice(self.nodes) for agent in self.agents} #randomly select initial positions for agents
        self.map = np.zeros((self.graph_size))
        for i in self.agents:
            self.map[self.agents_positions[i]] += 1
        
        self.agent_fov = np.zeros((self.agent_pop, 3))
        for i, agent in enumerate(self.agents):
            if self.agents_positions[agent] == 0:
                pos_minus = self.graph_size - 1
                self.agent_fov[i, 0] = self.map[pos_minus]
                self.agent_fov[i, 1] = self.map[0]
                self.agent_fov[i, 2] = self.map[1]
            elif self.agents_positions[agent] == self.graph_size - 1:
                pos_minus = self.graph_size - 2
                pos = self.graph_size - 1 
                self.agent_fov[i, 0] = self.map[pos_minus]
                self.agent_fov[i, 1] = self.map[pos]
                self.agent_fov[i, 2] = self.map[0]
            else:
                pos_minus = self.agents_positions[agent] - 1
                pos = self.agents_positions[agent]
                pos_plus = self.agents_positions[agent] + 1
                self.agent_fov[i, 0] = self.map[pos_minus]
                self.agent_fov[i, 1] = self.map[pos]
                self.agent_fov[i, 2] = self.map[pos_plus]

        self.state = {agent: self.map for agent in self.agents}

        if self.observability == 'full':
            observations = {agent: np.hstack((self.agents_positions[agent], self.map)) for agent in self.agents}     #how to add agent's position to the observation - add to map/fov?

        elif self.observability == 'partial':
            observations = {agent: np.hstack((self.agents_positions[agent], self.agent_fov[i])) for i, agent in enumerate(self.agents)}

        self.num_moves = 0

        return observations

    def step(self, actions):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        self.rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        if not actions:
            print('empty')
            self.agents = []
            return {}, {}, {}, {}, {}

        # rewards for all agents are placed in the .rewards dictionary
        # move agents to new positions
        for i in self.agents:
            if actions[i] == 0:
                if self.agents_positions[i] == 0:
                    self.agents_positions[i] = self.graph_size - 1
                else:
                    self.agents_positions[i] = self.agents_positions[i] - 1
            elif actions[i] == 1:
                if self.agents_positions[i] == self.graph_size - 1:
                    self.agents_positions[i] = 0
                else:
                    self.agents_positions[i] = self.agents_positions[i] + 1
            elif actions[i] == 2:
                self.agents_positions[i] = self.agents_positions[i]
            else:
                raise ValueError("Invalid action.")
            #reset and update map
            #can set back to 0 here as updating all at once - if sequantial would need to take away from previous location and add to new location
            self.map = np.zeros((self.graph_size))
            for i in self.agents:
                self.map[self.agents_positions[i]] += 1

            #update state
            self.state = {agent: self.map for agent in self.agents}
            
            #calculate updated field of view for each agent
            for i, agent in enumerate(self.agents):
                if self.agents_positions[agent] == 0:
                    self.agent_fov[i, 0] = self.map[self.graph_size - 1]
                    self.agent_fov[i, 1] = self.map[0]
                    self.agent_fov[i, 2] = self.map[1]
                elif self.agents_positions[agent] == self.graph_size - 1:
                    self.agent_fov[i, 0] = self.map[self.graph_size - 2]
                    self.agent_fov[i, 1] = self.map[self.graph_size - 1]
                    self.agent_fov[i, 2] = self.map[0]
                else:
                    self.agent_fov[i, 0] = self.map[self.agents_positions[agent] - 1]
                    self.agent_fov[i, 1] = self.map[self.agents_positions[agent]]
                    self.agent_fov[i, 2] = self.map[self.agents_positions[agent] + 1]
            
            #calculate rewards
            for i in self.agents:
                if self.map[self.agents_positions[i]] > 1:
                    self.rewards[i] = -1
                else:
                    self.rewards[i] = 1
            

            self.num_moves += 1
            # The truncations dictionary must be updated for all players.
            env_truncation = self.num_moves >= MAX_ITERS
            truncations = {agent: env_truncation for agent in self.agents}

            if env_truncation:
                self.agents = []

            # calculate observations for updated state
            if self.observability == 'full':
                observations = {agent: np.hstack((self.agents_positions[agent], self.map)) for agent in self.agents}
                #self.observations = {agent:{'pos': self.agents_positions[agent], 'map': self.map} for agent in self.agents}
            
            elif self.observability == 'partial':
                observations = {agent: np.hstack((self.agents_positions[agent],self.agent_fov[i])) for i, agent in enumerate(self.agents)}
                #self.observations = {agent:{'pos': self.agents_positions[agent], 'fov': self.agent_fov[i]} for i, agent in enumerate(self.agents)}

        if self.render_mode == "human":
            self.render()

        return observations, self.rewards, terminations, truncations, infos
