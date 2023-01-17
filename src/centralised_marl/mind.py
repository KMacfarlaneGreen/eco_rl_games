import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import random
import numpy as np
from zmq import device

class Mind:
    BATCH_SIZE = 256
    GAMMA = 0.98      #discount rate
    EPS_START = 0.9999   #exploration rate
    EPS_END = 0
    EPS_DECAY = 3000
    TAU = 0.05
    ALPHA = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, input_size, num_actions, lock, queue, destination = None, memory_length=1000000):
        self.network = DQN(input_size, num_actions)
        self.target_network = DQN(input_size, num_actions)
        self.lock = lock
        self.queue = queue        
        self.losses = []
        self.network.share_memory()      
        self.target_network.share_memory()

        self.input_size, self.num_actions = input_size, num_actions


        self.memory = ReplayMemory(memory_length)
        self.optimizer = optim.Adam(self.network.parameters(), self.ALPHA)
        self.steps_done = 0
        self.num_actions = num_actions

        self.target_network.load_state_dict(self.network.state_dict())   #is this the same as setting theta* = theta?
        self.input_size = input_size
        self.num_cpu = mp.cpu_count() // 2

    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.num_actions

    def get_losses(self):
        return self.losses

    def decide(self, state):
        #Output: action taken
        
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            np.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1 
        with torch.no_grad():
            state = torch.FloatTensor([state], device = self.device) #remove dependence on age and type
            q_values = self.network(state)                             #to do: add in this steps done attribute
            if sample > eps_threshold:
                return q_values.max(1)[1].view(1, 1).detach().item(), q_values.squeeze().tolist()
            else:
                rand = [[random.randrange(self.num_actions)]] # returns random choice of either 0 or 1 corresponding to possible actions
                return torch.tensor(rand, dtype=torch.long).detach().item(), q_values.squeeze().tolist()

    def decide_prob(self, state):
        with torch.no_grad():
                state = torch.FloatTensor([state], device = self.device) #remove dependence on age and type
                q_values = self.network(state)
                q_prob = (np.exp(q_values[0][0])/(np.exp(q_values[0][0])+np.exp(q_values[0][1])))
                action = np.random.binomial(1, q_prob)
        return action, q_values

    def remember(self, vals):     
        self.memory.push(vals)    #saves current state, action, next stae and reward to replay memory

    def opt(self, data):
        batch_state, batch_action, batch_next_state, batch_done, expected_q_values = data
        current_q_values = self.network(batch_state).gather(1, batch_action)
        #print('q vals',current_q_values)
        max_next_q_values = self.target_network(batch_next_state).detach().max(1)[0]

        for i, done in enumerate(batch_done):
            if done == [False]:
                expected_q_values[i] += (self.GAMMA * max_next_q_values[i].max())

        loss = F.mse_loss(current_q_values, expected_q_values.unsqueeze(1))   
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
        #for param in self.network.parameters():
            #param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        #queue.put(loss.item())
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.TAU * param.data + target_param.data * (1.0 - self.TAU))

        return loss

    def get_data(self):
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*transitions)
        batch_state = torch.cat([torch.FloatTensor(s) for s in batch_state]).view((self.BATCH_SIZE, 3))
        batch_action = torch.cat([torch.LongTensor(s) for s in batch_action]).view((self.BATCH_SIZE, 1))
        batch_reward = torch.cat([torch.FloatTensor(s) for s in batch_reward])
        batch_next_state = torch.cat([torch.FloatTensor(s) for s in batch_next_state])

        #print('state size', batch_state.size(), 'action size', batch_action.size())
        #print('state [1]', batch_state[0], 'action 1', batch_action[0])

        expected_q_values = batch_reward
        return (batch_state, batch_action, batch_next_state, batch_done, expected_q_values)

    def train(self):
        if len(self.memory) < self.BATCH_SIZE:
            return 1

        data = self.get_data()
        loss = self.opt(data)
        self.losses.append(loss.detach().numpy())
        #processes = []
        #for _ in range(self.num_cpu):     
            #data = self.get_data()
            #p = mp.Process(target=self.opt, args=(data, self.lock, self.queue))
            #p.start()
            #processes.append(p)
        #for p in processes:
            #loss = self.queue.get() # will block
            #self.losses.append(loss)
        #for p in processes:
            #p.join()

        return 0


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, nS, nA):        #statespace size, neurons in hidden layer, actionspace size
        self.nH = 16
        super(DQN, self).__init__()
        self.l1 = nn.Linear(nS, self.nH) # 3
        self.out = nn.Linear(self.nH, nA)
        for m in self.modules():       
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')     #if this initialises weights for q and q* then they won't be the same? Should the initialisation be random?


    def forward(self, x):
        x = F.relu((self.l1(x)))
        #x = x.mean(-1).mean(-1)
        #x = torch.cat([x], dim=1)
        out = self.out(x)
        #x= F.softmax(self.out(x), dim = 1)
        return F.relu(out)

