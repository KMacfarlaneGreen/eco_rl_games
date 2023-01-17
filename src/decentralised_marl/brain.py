#decentralised code adapted from https://github.com/mohammadasghari/dqn-multi-agent-rl/blob/master

# to do: convert from tensorflow to pytorch
#        change structure of NN to be consistent with previous experiments


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import random
import numpy as np
from zmq import device

# check arguments actually called in the functions 

TAU = 0.05
HUBER_LOSS_DELTA = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Brain(object):

    def __init__(self, state_size, action_size, brain_name, arguments):
        self.state_size = state_size
        self.action_size = action_size
        self.weight_backup = brain_name
        self.batch_size = arguments['batch_size']
        self.learning_rate = arguments['learning_rate']
        self.test = arguments['test']
        #self.num_nodes = arguments['number_nodes']
        #self.dueling = arguments['dueling']
        #self.optimizer_model = arguments['optimizer']
        
        self.model = DQN(state_size, action_size)
        self.model_ = DQN(state_size, action_size)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model_.load_state_dict(self.model.state_dict())

        #self.queue = queue        
        self.losses = []

        self.num_cpu = mp.cpu_count() // 2
 
    def opt(self, x, y):
        x = torch.Tensor(x).requires_grad_()
        print(x)
        y = torch.Tensor(y).requires_grad_()
        print(y)
        loss = F.mse_loss(x, y)  
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        #for param in self.model.parameters():
          #print('param',param)
          #param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        return loss

        #queue.put(loss.item())

    #def train(self, x, y):
        #processes = []
        #for _ in range(self.num_cpu):     
            #p = mp.Process(target=self.opt, args=([x,y], self.queue))  #x is the state and y is the target
            #p.start()
            #processes.append(p)
        #for p in processes:
            #loss = self.queue.get() # will the queue still work in decentralised framework or need to change this?
            #self.losses.append(loss)
       # for p in processes:
            #p.join()

        #return 0

    def predict(self, state, target=False):
        state = torch.Tensor(state)
        print(state.dtype)
        if target:  # get prediction from target network
            prediction_ = self.model_(state)
            return prediction_.detach().numpy()
        else:  # get prediction from local network
            prediction = self.model(state)
            return prediction.detach().numpy()

    def predict_one_sample(self, state, target=False):
        return self.predict(state.reshape(1,self.state_size), target=target).flatten()

    def update_target_model(self):

        # update weights in target network
        for target_param, param in zip(self.model_.parameters(), self.model.parameters()):
            target_param.data.copy_(TAU * param.data + target_param.data * (1.0 - TAU))

    def save_model(self):
        torch.save(self.model.state_dict(),self.weight_backup)   #saves weights - want to save losses as well?


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
        out = self.out(x)
        return F.relu(out)


