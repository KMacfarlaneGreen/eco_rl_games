import numpy as np
import networkx as nx

def create_lattice_graph(dim1, dim2):
    '''Create the lattice graph environment'''
    graph = nx.grid_2d_graph(dim1, dim2)
    nodes = list(graph.nodes)

    return graph, nodes

def move_up(curr_pos, max_node):
    '''Define action move up'''
    new_pos = np.copy(curr_pos)
    
    if new_pos[1] < max_node:
        
        new_pos[1] = new_pos[1] + 1
    #else:
        #new_pos[1] = new_pos[1]
    
    return new_pos

def move_down(curr_pos):
    '''Define action move down'''
    new_pos = np.copy(curr_pos)
    
    if new_pos[1] != 0.0:
        
        #new_pos[1] = new_pos[1]
    #else:
        new_pos[1] = new_pos[1] - 1
    
    return new_pos

def move_right(curr_pos, max_node):
    '''Define action move right'''
    new_pos = np.copy(curr_pos)
    
    if new_pos[0] < max_node:
        
        new_pos[0] = curr_pos[0] + 1
    #else:
        #new_pos[0] = new_pos[0]
    
    return new_pos

def move_left(curr_pos):
    '''Define action move left'''
    new_pos = np.copy(curr_pos)
    
    if new_pos[0] != 0.0:
        
        #new_pos[0] = new_pos[0]
    #else:
        new_pos[0] = new_pos[0] - 1
    
    return new_pos

def stay(curr_pos):
    '''Define action to stay still'''
    new_pos = np.copy(curr_pos)
    
    return new_pos

def initialise_rew(N, Nt):
    '''initialise stored arrays for rewards and actions'''
    
    rewards = np.zeros((N,Nt))
    actions = np.zeros((N,Nt))
    
    return rewards, actions

def initialise_params(N, Nt):
    '''initialise parameter values for agents'''
    
    theta_a = np.zeros(N) #do these need to be stored for every time step or updated in place?
    alpha_a = np.zeros(N)  #initially try updating them in place   #0.9
    k_a = np.ones(N)*0.1 #np.zeros(N)      #0.1
    
    for n in range(0,N):
        
        theta_a[n] = np.random.random()
        alpha_a[n] = np.random.random()
        #k_a[n] = np.random.randint(0,2)
        
    return theta_a, alpha_a, k_a  

def survival(N_i, alpha_a, k_a):
    '''Calculating whether agents die or survive'''
    lamda = np.zeros(N_i)
    p = np.zeros(N_i)
    
    for i in range(0,N_i):
        
        p[i] = alpha_a[i] ** k_a[i]
        
        if p[i] > 1:
            
            p[i] = 1
        
        if np.random.binomial(1, p[i]) == 1:
            lamda[i] = 1
        else:
            lamda[i] = 0
            
    return lamda, p

def survival_at_node(lamda, N_i):
    '''returns how many agents survive at a given node'''
    s = 0
    
    for n in N_i:
        
        s += lamda[n]
        
    return s

def catch(N_i, theta_a):
    '''Determines whether agents successfully consume food'''
    z = np.zeros(N_i)
    
    for i in range(0,N_i):
        
        theta = theta_a[i]
        
        if theta > 1:
            
            theta = 1
        
        if np.random.binomial(1, theta)== 1:
            z[i] = 1
        else:
            z[i] = 0
            
    return z

def beta(z, N_i):
    '''Returns how many agents on a given node consume food'''
    beta = 0
    
    for n in N_i:
        
        beta += z[n]
        
    return beta

def update_pos(pos, l, r, u, d, s):       #to do: change probabilities to vector
    '''Updates agents position according to movement probabilities. Returns new position and action'''
    new_pos = np.copy(pos)
    
    A = np.random.multinomial(50, [l,r,u,d,s])
    
    a = np.max(A)
    
    if a == A[0]:
        new_pos = move_left(pos)
        action = 2
    
    if a == A[1]:
        new_pos = move_right(pos,3)
        action = 1
        
    if a == A[2]:
        new_pos = move_up(pos,9)
        action = 3
        
    if a == A[3]:
        new_pos = move_down(pos)
        action = 4
        
    if a == A[4]:
        new_pos = stay(pos)
        action = 0
            
    return new_pos, action

def update_pos_stay(pos):
    '''position definitely stays the same'''
    
    action = 0
    
    return pos, action 

def update_k(k_a, reward, lamda, N_i, time):
    '''Updates agent's count parameter depending on the reward it receives'''
    for n in range(0,N_i):
        
        if reward[n][time] == 1.0:     
            
            k_a[n] = k_a[n] - 0.1    
            
        if reward[n][time] == -1.0:
            
            k_a[n] = k_a[n] + 0.5        #this update defines how quickly agents die if they don't consume food 
            
        if reward[n][time] == -5.0:
            
            k_a[n] = k_a[n] + 10
        
        if k_a[n] < 0:
            k_a[n] = 0.1 
            
    return k_a

def update_theta(theta_a, reward, lamda, N_i, time):
    '''Updates agent's fitness parameter depending on the reward it receives'''
    for n in range(0,N_i):
        
        if theta_a[n] < 1.0:
        
            if reward[n][time] == 1.0:
            
                theta_a[n] = theta_a[n] + 0.01    #try adjusting these hyperparameters also

            #if reward[n][time] == -1.0:
                #theta_a[n] = theta_a[n] - 0.01

    return theta_a

def coords_to_num(trajectories_n_i):
    '''Converts coordinate location to node number'''
    loc_init_zero = str(int(trajectories_n_i[0]))
    loc_init_one = str(int(trajectories_n_i[1]))
        
    if int(loc_init_zero) == 0:
        part_num = loc_init_one
    else:
        part_num = loc_init_zero + loc_init_one
            
    return part_num

