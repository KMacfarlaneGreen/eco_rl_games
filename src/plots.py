import imageio
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import networkx as nx
import os

def save_run(particles_at_nodes, resources_at_nodes, rewards, savename):
    total_preds = np.sum(particles_at_nodes, axis = 0)
    total_prey = np.sum(resources_at_nodes, axis = 0) 
    rewards = np.sum(rewards, axis=0)

    preds = open('results/preds_{}.csv'.format(savename),'a')
    prey = open('results/prey_{}.csv'.format(savename),'a')
    rews = open('results/rewards_{}.csv'.format(savename),'a')

    np.savetxt(preds, total_preds, fmt='%1.3f', newline=", ")
    preds.write("\n")
    np.savetxt(prey, total_prey, fmt='%1.3f', newline=", ")
    prey.write("\n")
    np.savetxt(rews,rewards , fmt='%1.3f', newline=", ")
    preds.write("\n")
    prey.write("\n")
    rews.write("\n")
    preds.close()
    prey.close()
    rews.close()
    #np.savetxt('preds_{}.csv'.format(savename), total_preds, delimiter=',', fmt='%s')
    #np.savetxt('prey_{}.csv'.format(savename), total_prey, delimiter=',', fmt='%s')
    #np.savetxt('rewards_{}.csv'.format(savename), rewards, delimiter=',', fmt='%s')

    return

#change to functions
def plot_dynamics(particles_at_nodes, resources_at_nodes, Nt, savename):

    total_preds = np.sum(particles_at_nodes, axis = 0)
    total_prey = np.sum(resources_at_nodes, axis = 0)   

    fig = plt.figure()
    time_sim = np.linspace(0,Nt,Nt)

    plt.plot(time_sim[0:Nt], total_preds[0:Nt], label = 'predators')
    plt.plot(time_sim[0:Nt], total_prey[0:Nt], label = 'prey')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Population')

    plt.show()

    filename = '{}.png'.format(savename)
    
    fig.savefig(filename)

def plot_average_dynamics(all_preds, all_prey, Nt, savename):

    tot_preds= np.sum(np.array(all_preds), axis = 0)
    tot_prey = np.sum(np.array(all_prey), axis = 0)

    avg_preds = tot_preds[0:100]/10
    avg_prey = tot_prey[0:100]/10   

    fig = plt.figure()
    time_sim = np.linspace(0,Nt,Nt)

    plt.plot(time_sim[0:Nt], avg_preds[0:Nt], label = 'predators')
    plt.plot(time_sim[0:Nt], avg_prey[0:Nt], label = 'prey')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Population')

    plt.show()

    filename = '{}.png'.format(savename)
    
    fig.savefig(filename)

def plot_gif(G, Nt, N, resources_at_nodes, particles_at_nodes, trajectories, experiment):
    filenames = []

    for i in range(0,Nt):
        #print('iteration',i)
        fig, (ax1) = plt.subplots(1,figsize =(15,5))
        node_pos = dict((n, n) for n in G.nodes())
        nx.draw_networkx(G, with_labels = False, pos=node_pos, ax = ax1, node_size = 10, node_color = 'white', width =0.5)
        nodes = list(G.nodes)
        for v in range(0,len(nodes)):
            resources = int(resources_at_nodes[v][i])
            #print(resources)
            ax1.plot(nodes[v][0],nodes[v][1], color = 'orange', marker = 'o', markersize = 2*resources )
        for j in range(0,len(nodes)):
        #for n in range(0,N):
                #node_loc = trajectories[n][i]  #per agent
            
            pop = int(particles_at_nodes[j][i])   #per node
            node_loc = nodes[j]
            #print(node_loc, pop)
            ax1.plot(node_loc[0],node_loc[1], color = 'red', marker = 'o', markersize = 2*pop)
        
        plt.title(f'{i} Iteration')


        ax1.set_xticks([])
        ax1.set_yticks([])
        #plt.show()
    
        filename = f'{i}.png'
        filenames.append(filename)
    
        # save frame
        fig.savefig(filename)

        # build gif
    with imageio.get_writer('huffmites{}.gif'.format(experiment), mode='I', duration = 0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        
    # Remove files
    for filename in set(filenames):
        os.remove(filename)

#show gif
def show_gif(gif):

    img = mpimg.imread(gif)
    plt.imshow(img)

