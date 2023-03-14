import gymnasium 
import numpy as np
import random

from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent


LEFT = 0
RIGHT = 1
STAY = 2
MOVES = ['LEFT', 'RIGHT', 'STAY']
MAX_ITERS = 100

def make_multiagent(env_name_or_creator):
    return make_multi_agent(env_name_or_creator)

class AntisocialRingEnv(MultiAgentEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config):
        super().__init__()
        self.num_agents = config["num_agents"]
        self.agents = [str(i) for i in range(self.num_agents)]
        self.graph_size = config["graph_size"]
        self.nodes = [i for i in range(self.graph_size)]
        self._agent_ids = set(self.agents)
        self.terminateds = set()
        self.truncateds = set()
        self.observation_space = gymnasium.spaces.Box(low=np.zeros((3)), high = self.graph_size*np.ones((3)), dtype=np.float32)
        self.action_space = gymnasium.spaces.Discrete(3)
        self.resetted = False

    def render(self, mode='human'):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        for i in range(self.num_agents):
            print("Agent {} is at position {} and recieved reward {}".format(i, self.agents_positions[i], self.rewards[self.agents[i]]))

    
    def reset(self,*,seed = None, options=None):
        #super().reset(seed = seed)
        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()
        infos = {agent: {} for agent in self.agents}
        
        self.agents_positions = {agent: np.random.choice(self.nodes) for agent in self.agents} #randomly select initial positions for agents
        self.map = np.zeros((self.graph_size))
        for i in self.agents:
            self.map[self.agents_positions[i]] += 1

        self.agent_fov = np.zeros((self.num_agents, 3))
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
        self.num_moves = 0
        observations = {agent: self.agent_fov[i] for i, agent in enumerate(self.agents)}
        return observations, infos

    def step(self, actions):
        #print(self.agents)
        self.rewards = {agent: 0 for agent in self.agents}
        #print(self.rewards)
        terminations = {agent: False for agent in self.agents}
        #print(terminations)
        infos = {agent: {} for agent in self.agents}
        #print(infos)
        truncated = {agent:{} for agent in self.agents}
        #print(truncated)
        #print(actions)
        # rewards for all agents are placed in the .rewards dictionary
        # move agents to new positions
        for i, action in actions.items():
          if action == 0:
              if self.agents_positions[i] == 0:
                  self.agents_positions[i] = self.graph_size - 1
              else:
                  self.agents_positions[i] = self.agents_positions[i] - 1
          elif action == 1:
              if self.agents_positions[i] == self.graph_size - 1:
                  self.agents_positions[i] = 0
              else:
                  self.agents_positions[i] = self.agents_positions[i] + 1
          elif action == 2:
              self.agents_positions[i] = self.agents_positions[i]
          else:
              raise ValueError("Invalid action.")
            #reset and update map
            #can set back to 0 here as updating all at once - if sequantial would need to take away from previous location and add to new location
          self.map = np.zeros((self.graph_size))
          for i in self.agents:
              self.map[self.agents_positions[i]] += 1

        #update agent fov
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
          #print(self.rewards)
          observations = {agent: self.agent_fov[i] for i, agent in enumerate(self.agents)}
              #self.observations = {agent:{'pos': self.agents_positions[agent], 'fov': self.agent_fov[i]} for i, agent in enumerate(self.agents)}
          #print(observations)
          if self.render_mode == "human":
              self.render()

          self.num_moves += 1
          if self.num_moves >= MAX_ITERS:
            terminations["__all__"] = True
            truncated["__all__"] = True
          else:
            truncated["__all__"] = False
            terminations["__all__"] = False
          #print(observations,self.rewards, terminations, truncated,infos)
          return observations, self.rewards, terminations,truncated, infos