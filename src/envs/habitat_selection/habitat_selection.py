import functools
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Discrete, Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

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
    metadata = {'render.modes':['human'], 'name': 'habitat_selection_v0'}

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
        self.agent_pop = 5   #e.g. 5 agents, 20 nodes
        self.graph_size = 20
        self.state_size = self.graph_size * 2  
        self.nodes = [i for i in range(self.graph_size)]
        self.quality = np.random.randint(0, 10, self.graph_size)  #assign quality to each node
        self.render_mode = render_mode
        self.possible_agents = [str(i) for i in range(self.agent_pop)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(self.agent_pop))))

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=np.zeros((1 + 2*self.graph_size)), high = 10*np.ones((1 + 2*self.graph_size)), dtype=np.float32) #agent's current node, quality of each node, and number of agents at each node
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.graph_size)    #agents select a node to move to 

    def seed(self, seed=None):
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
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed = None, options = None): 
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.agents_positions = {agent: np.random.choice(self.nodes) for agent in self.agents} #randomly assign agents to nodes
        self.map = np.zeros((self.graph_size))
        for agent in self.agents:
            self.map[self.agents_positions[agent]] += 1
        
        observations = {agent: np.hstack((self.agents_positions[agent], self.quality, self.map)) for agent in self.agents}

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

        for agent in self.agents:
            if actions[agent] in self.nodes:
                self.map[self.agents_positions[agent]] -= 1
                self.agents_positions[agent] = actions[agent]
                self.map[self.agents_positions[agent]] += 1
                self.rewards[agent] = self.quality[self.agents_positions[agent]] - self.map[self.agents_positions[agent]]
            else:
                print('invalid action')
        
        self.num_moves += 1
        env_truncation = self.num_moves >= MAX_ITERS
        truncations = {agent: env_truncation for agent in self.agents}

        if env_truncation:
            self.agents = []

        observations = {agent: np.hstack((self.agents_positions[agent], self.quality, self.map)) for agent in self.agents}

        if self.render_mode == "human":
            self.render()
        
        return observations, self.rewards, terminations, truncations, infos

    