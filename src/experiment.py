from ring_environment import Environment

from itertools import count
from multiprocessing import Process, Lock

import time
import random
import os, sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import numpy as np

def play(map, episodes, iterations, eps=1e-6):
    #to do:edit so can run with my environment
    # map.configure(prey_reward, stuck_penalty, agent_max_age)
    agents = map.get_agents()
    #print(agents)
    times = 0
    for episode in range(episodes):
        c = 0
        for t in count():
            t_start = time.time()
            state = map.get_map()
            random.shuffle(agents)

            keys = ["tot_reward"]
            rews = {key: 0 for key in keys}       #change from dict to list?
            #print(rews)
            counts = {key: 0 for key in keys}
            #print(counts)
            for agent in agents:
                agent_id = agent.get_id()
                #print(agent_id)
                agent_state = agent.get_state()
                #print('state/no agents on node',agent_state)
                action = agent.decide(agent_state)
                #print('action',action)
                rew = map.step(agent, action)
                #print('reward',rew)
                rews["tot_reward"] += rew     #name = A, B, prey 
                counts["tot_reward"] += 1
                #print('cumulative reward',rews)
            
            map.record(rews)

            map.update()

            next_state = map.get_map()

            time_elapsed = time.time() - t_start
            times += time_elapsed
            #print('times', times)
            avg_time = times / (t + 1)
            print("I: %d\tTime Elapsed: %.2f" % (t+1, avg_time), end='\r')
            if abs(next_state - state).sum() < eps:
                c += 1
            #print('c', c)

            if t == (iterations - 1) or c == 20:
                break
            #print('t',t)

            state = next_state
        map.save(episode)
    print("SIMULATION IS FINISHED.")


if __name__ == '__main__':
    [_, name, iterations] = sys.argv

    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)

    episodes = 1
    iterations = int(iterations)
    l = Lock()

    args = ["Name"]  #change to ring env args

    society = Environment

    play(society(100, num_actions = 2, name=name, max_iteration = int(iterations), num_agents = 20,
        lock=l), 1, iterations) #check ordering