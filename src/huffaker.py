import numpy as np
from src.environment import * #change to classes

def initialise_nodes_experiment1(N, Nt, nodes):
    '''initialise arrays for particle trajectories, number of particles at each node and resources at each node for experiment 1'''
    trajectories =  np.zeros((N,Nt,2))
    particles_at_nodes = np.zeros((len(nodes),Nt))    #number of particles/resource at node 0-99 at each time t
    resources_at_nodes = np.zeros((len(nodes),Nt))
    
    for i in [10,11,20,21] :
        resources_at_nodes[i][0] = 10
    
    for n in range(0,N):      #for each particle
        trajectories[n][0] = nodes[10] #initialise node locations at same node as resources
        loc_init = 10
        particles_at_nodes[loc_init][0] +=1
        
    return trajectories, particles_at_nodes, resources_at_nodes

def initialise_nodes_experiment2(N, Nt, nodes):
    '''resource distribution for experiment 2'''
    trajectories =  np.zeros((N,Nt,2))
    particles_at_nodes = np.zeros((len(nodes),Nt))    #number of particles/resource at node 0-99 at each time t
    resources_at_nodes = np.zeros((len(nodes),Nt))
    #want total resource of 40 evenly distributed across every other node

    for i in range(0, len(nodes), 2):
        resources_at_nodes[i][0] = 2

    for n in range(0,N):      #for each particle
        trajectories[n][0] = nodes[10] #initialise node locations at same node 
        loc_init = 10
        particles_at_nodes[loc_init][0] +=1

    return trajectories, particles_at_nodes, resources_at_nodes



def play(N, Nt,l,r,u,d,s, g, init_func,dim1,dim2): 
            #change probabilities to vector
    graph, nodes = create_lattice_graph(dim1, dim2)
    trajectories, particles_at_nodes, resources_at_nodes = init_func(N, Nt, nodes) 
    rewards, action = initialise_rew(N, Nt)
    theta_a, alpha_a, k_a = initialise_params(N,Nt)
    prob = []

    for i in range(1,Nt):    #for each time step change Nt=2 for 1 timestep
        agent_locs = np.zeros(N)
        z = catch(N, theta_a)
        #print(z)
        lamda, p = survival(N, alpha_a, k_a)
        print(lamda)
        prob.append(p)
   
        for n in range(0,N):    #for each particle
        
            if lamda[n] == 1.0:    #only update position if agent survives 
                
                trajectories[n][i], action[n][i] = update_pos(trajectories[n][i-1],l,r,u,d,s)   #update position (action)
                
                part_num = coords_to_num(trajectories[n][i])
    
                particles_at_nodes[int(part_num)][i] +=1
            
                agent_locs[n] = int(part_num)
     
            if lamda[n] == 0.0:
            
                trajectories[n][i] = trajectories[n][i-1]  #set trajectory to remain the same if agent dies
        
                part_num = coords_to_num(trajectories[n][i])
            
                agent_locs[n] = int(part_num)
            
                rewards[n][i] = -5   #add large negative reward if agent dies
                
        for v in range(0,len(nodes)):       #for each node
            
            n_a_v = particles_at_nodes[v][i]  #total preds/prey at each node
            n_b_v = resources_at_nodes[v][i-1]
            #print('resource', n_b_v) 
            #print('agents', n_a_v)  
        
            agents_node_v =[]
        
            for j in range(0,N):
            
                if int(agent_locs[j]) == v and lamda[j] == 1.0:  #n
                
                    agents_node_v.append(j)         #gives which agents are located on that node
                
            #print('agents at node', agents_node_v)  
            b = beta(z, agents_node_v)          #calculate beta for the agents located on node v 
            #print('consumption',b)
            for x in agents_node_v:
                
                if b <= n_b_v and b > 0:

                    if z[x] == 1.0:
                
                        rewards[x][i] = 1         #agents are rewarded if they survive, successfully catch prey and do not exceed the available resource
                    
                if z[x] == 0.0:
                    
                    rewards[x][i] = -1
          
            #nbv update  
            resources_at_nodes[v][i] = n_b_v *np.exp(g*i) - b #resource or prey increases exponentially dependent on the growth rate
        
            if resources_at_nodes[v][i]<0:        #cannot be negative
                resources_at_nodes[v][i] = 0
        
        #param update
        
        k_a = update_k(k_a, rewards, lamda, N, i)
        theta_a = update_theta(theta_a, rewards, lamda, N, i)
    
    #if not running in notebook results need to be saved to e.g. csv file. To do

    return graph, particles_at_nodes, resources_at_nodes, trajectories, rewards




# Simulation parameters
N         = 20    # Number of particles 
t         = 0      # current time of the simulation
tEnd      = 100.0   # time at which simulation ends
dt        = 1 #0.01 
# number of timesteps
Nt = int(np.ceil(tEnd/dt))
l = 0.2   #move left, right, up or down with 20% probability - think about how to update these (per agent)
r = 0.2
u = 0.2
d = 0.2
s = 0.2
#prey growth rate
g = 0 #0.001
dim1 = 4
dim2 = 10

#particles_at_nodes, resources_at_nodes, trajectories, rewards = play(N, Nt, dim1,dim2, l,r,u,d,s,g, initialise_nodes_experiment1)