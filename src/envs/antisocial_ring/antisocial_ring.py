import functools
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

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

class raw_env(AECEnv):
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
        self.state_size = self.graph_size   #how to represent state?
        #self.agents_positions = []
        self.nodes = [i for i in range(self.graph_size)]
        #self.positions_idx = []
        #self.num_episodes = 0
        #self.terminal = False
        self.render_mode = render_mode
        self.observability = observability
        self.possible_agents = [str(i) for i in range(self.agent_pop)]
        #self.agent_order = [str(i) for i in range(self.num_agents)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(self.agent_pop))))
        #self.agent_selection = agent_selector(self.agent_order)
        self._action_spaces = {agent: Discrete(3) for agent in self.possible_agents}
        if self.observability == 'full':
            self._observation_spaces = {agent: Box(low=np.zeros((self.graph_size)), high = self.agent_pop*np.ones((self.graph_size)),dtype=np.float32) for agent in self.possible_agents}
        elif self.observability == 'partial':
            self._observation_spaces = {agent: Box(low=np.zeros((3)), high = self.agent_pop*np.ones((3)),dtype=np.float32) for agent in self.possible_agents}
        #self.rewards = {agent: 0 for agent in self.agents}
        #self.dones = {agent: False for agent in self.agents}
        #self.infos = {agent: {} for agent in self.agents}
        #self.viewer = None

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if self.observability == 'full':
            return Box(low=np.zeros((self.graph_size)), high = self.agent_pop*np.ones((self.graph_size)), dtype=np.float32)
        elif self.observability == 'partial':
            return Box(low=np.zeros((3)), high = self.agent_pop*np.ones((3)), dtype=np.float32)

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

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent], dtype = np.float32)

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
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.actions = {agent: None for agent in self.agents}

        self.agents_positions = {agent: 0 for agent in self.agents}
        for i in self.agents:
          self.agents_positions[i] = np.random.choice(self.nodes)  #randomly select initial positions for agents
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
            self.observations = {agent: self.map for agent in self.agents}

        elif self.observability == 'partial':
            self.observations = {agent: self.agent_fov[i] for i, agent in enumerate(self.agents)}

        self.num_moves = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
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
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # save action of selected agent 
        self.actions[self.agent_selection] = action  

        # collect reward if it is the last agent to act - want to collect rewards at the end or as step through?
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            # move agents to new positions
            for i in self.agents:
                if self.actions[i] == 0:
                    if self.agents_positions[i] == 0:
                        self.agents_positions[i] = self.graph_size - 1
                    else:
                        self.agents_positions[i] = self.agents_positions[i] - 1
                elif self.actions[i] == 1:
                    if self.agents_positions[i] == self.graph_size - 1:
                        self.agents_positions[i] = 0
                    else:
                        self.agents_positions[i] = self.agents_positions[i] + 1
                elif self.actions[i] == 2:
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
            self.truncations = {
                agent: self.num_moves >= MAX_ITERS for agent in self.agents
            }

            # calculate observations for updated state
            if self.observability == 'full':
                self.observations = {agent: self.map for agent in self.agents}
            
            elif self.observability == 'partial':
                self.observations = {agent: self.agent_fov[i] for i, agent in enumerate(self.agents)}

        else:
            # should the state and observations update each step or only at the end?
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()