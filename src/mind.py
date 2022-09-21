import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

import random
import numpy as np

class Mind:
    BATCH_SIZE = 256
    GAMMA = 0.98
    EPS_START = 0.9999
    EPS_END = 0
    EPS_DECAY = 100000
    TAU = 0.05
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, input_size, num_actions, destination = None, memory_length=1000000):
        #self.network = DQN(input_size, num_actions)
        #self.target_network = DQN(input_size, num_actions)
        #self.lock = lock
        #self.queue = queue
        self.losses = []
        #self.network.share_memory()
        #self.target_network.share_memory()

        self.input_size, self.num_actions = input_size, num_actions


        #self.memory = ReplayMemory(memory_length)
        #self.optimizer = optim.Adam(self.network.parameters(), 0.001)
        #self.steps_done = 0
        #self.num_actions = num_actions

        #self.target_network.load_state_dict(self.network.state_dict())
        #self.input_size = input_size
        #self.num_cpu = mp.cpu_count() // 2

    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.num_actions

    def get_losses(self):
        return self.losses

    def decide(self, state):
        #what do I want the inputs and outputs of this decide function to be?
        #Output: vector of probabilities for each agent's movement decision? or actual actions they take?
        
        #sample = random.random()
        #eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            #np.exp(-1. * self.steps_done / self.EPS_DECAY)
        #self.steps_done += 1                              #to do: add in this steps done attribute
        #if sample > eps_threshold:
            #with torch.no_grad():
                #state = torch.FloatTensor([[state]], device=self.device)
                ##age = torch.FloatTensor([[age]], device=self.device)
                #q_values = self.network(type * state, age)
                #return q_values.max(1)[1].view(1, 1).detach().item()
        rand = [[random.randrange(self.num_actions)]]
        #currently returns random choice of either 0 or 1 corresponding to possible actions
        #therefore I need to change this so that this choice depends on the agent's movement probability
        #this probability will then be parametrised and updated for optimisation
        #although current set up is equivalent to p=0.5
        return torch.tensor(rand, device=self.device, dtype=torch.long).detach().item()