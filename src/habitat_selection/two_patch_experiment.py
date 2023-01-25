import numpy as np
import os
import argparse
import pandas as pd
from two_patch_env import Two_patch_selection
from agent import Agent
from brain import Brain

ARG_LIST = ['learning_rate', 'memory_capacity', 'batch_size', 'target_frequency', 'maximum_exploration',
            'max_timestep', 'first_step_memory', 'replay_steps', 'memory', 'agents_number', 'graph_size']

def get_name_brain(args):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    return './habitat_selection/weights_files/' + file_name_str + '_'  + '.h5'

def get_loss_name(args, idx):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])
    return './habitat_selection/losses/' + file_name_str + '_' + str(idx) + '.csv'
    
def get_name_rewards(args):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    return './habitat_selection/rewards_files/' + file_name_str + '.csv'


def get_name_timesteps(args):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    return './habitat_selection/timesteps_files/' + file_name_str + '.csv'

def get_name_actions(args):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])
    #os.makedirs('./results_ring_antisocial/timesteps_files/'+ file_name_str + '.csv')
    return './habitat_selection/actions/' + file_name_str + '.csv'

def get_name_states(args):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])
    #os.makedirs('./results_ring_antisocial/timesteps_files/'+ file_name_str + '.csv')
    return './habitat_selection/states/' + file_name_str + '.csv'

def get_name_rews(args):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])
    #os.makedirs('./results_ring_antisocial/timesteps_files/'+ file_name_str + '.csv')
    return './habitat_selection/rewards_timesteps/' + file_name_str + '.csv'



class Environment(object):

    def __init__(self, arguments):
        current_path = os.path.dirname(__file__)  # Where your .py file is located
        self.env = Two_patch_selection(arguments, current_path)
        self.episodes_number = arguments['episode_number']
        self.max_ts = arguments['max_timestep']
        self.test = arguments['test']
        self.filling_steps = arguments['first_step_memory']
        self.steps_b_updates = arguments['replay_steps']
        self.max_random_moves = arguments['max_random_moves']

        self.num_agents = arguments['agents_number']
        self.graph_size = arguments['graph_size']


    def run(self, agents, brain, file1, file2, file3, file4, file5):

        total_step = 0
        rewards_list = []
        timesteps_list = []
        max_score = -10000     #check this
        for episode_num in range(self.episodes_number):
            state = self.env.reset()

            # converting list of positions to an array
            state = np.array(state)
            state = state.ravel()

            done = False
            reward_all = 0
            time_step = 0
            while not done and time_step < self.max_ts:
                actions = []   
                for agent in agents:
                    actions.append(agent.greedy_actor(state))  #list of action of each agent at each time step
                next_state, rewards, done = self.env.step(actions)  #needs to give next state,reward for each of the actions i.e. [len(num_agents)]
                # converting list of positions to an array
                next_state = np.array(next_state)
                next_state = next_state.ravel()

                with open(file3, "a") as f:
                  f.write("%s, %s\n" % (time_step, actions)) 
              
                with open(file4, "a") as f:
                  f.write("%s, %s\n" % (time_step, state)) 
                
                with open(file5, "a") as f:
                  f.write("%s, %s\n" % (time_step, rewards))

                if not self.test:
                    for agent in agents:
                        loss_list = []
                        agent.observe((state, actions, rewards, next_state, done))  #pushing to replay memory 
                        if total_step >= self.filling_steps:
                            agent.decay_epsilon()
                            if time_step % self.steps_b_updates == 0:
                                loss = agent.replay()     #this is the training step
                            agent.update_target_model()
                            loss_list.append(loss)
                        idx = agent.get_index()
                        loss_file = get_loss_name(args,idx)
                        df_loss = pd.DataFrame(loss_list, columns=None, index=[f'{time_step}'])
                        df_loss.to_csv(loss_file, mode = 'a')

                total_step += 1
                time_step += 1
                state = next_state     
                reward_all += sum(rewards)  #total rewards for all agents


            rewards_list.append(reward_all)
            timesteps_list.append(time_step)

            print("Episode {p}, Score: {s}, Final Step: {t}, Goal: {g}".format(p=episode_num, s=reward_all,
                                                                               t=time_step, g=done))

            if not self.test:
                if episode_num % 100 == 0:
                    df = pd.DataFrame(rewards_list, columns=['score'])
                    df.to_csv(file1)

                    df = pd.DataFrame(timesteps_list, columns=['steps'])
                    df.to_csv(file2)

                    if total_step >= self.filling_steps:
                        if reward_all > max_score:
                            #for agent in agents:
                            brain.save_model()
                            max_score = reward_all

if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    # DQN Parameters
    parser.add_argument('-e', '--episode-number', default=1000000, type=int, help='Number of episodes')  #how have they defined episodes here?
    parser.add_argument('-l', '--learning-rate', default=0.00005, type=float, help='Learning rate')
    parser.add_argument('-m', '--memory-capacity', default=1000000, type=int, help='Memory capacity')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('-t', '--target-frequency', default=10000, type=int,
                        help='Number of steps between the updates of target network')
    parser.add_argument('-x', '--maximum-exploration', default=100000, type=int, help='Maximum exploration step')
    parser.add_argument('-fsm', '--first-step-memory', default=0, type=float,
                        help='Number of initial steps for just filling the memory')
    parser.add_argument('-rs', '--replay-steps', default=4, type=float, help='Steps between updating the network')
    parser.add_argument('-mt', '--memory', choices=['UER'], default='UER')
    parser.add_argument('-gn', '--gpu-num', default='2', type=str, help='Number of GPU to use')
    parser.add_argument('-test', '--test', action='store_true', help='Enable the test phase if "store_false"')

    # Game Parameters
    parser.add_argument('-k', '--agents-number', default=5, type=int, help='The number of agents')
    parser.add_argument('-g', '--graph-size', default=10, type=int, help='Grid size')
    parser.add_argument('-ts', '--max-timestep', default=100, type=int, help='Maximum number of timesteps per episode')
    parser.add_argument('-rm', '--max-random-moves', default=0, type=int,
                        help='Maximum number of random initial moves for the agents')

    args = vars(parser.parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_num']

    env = Environment(args) 

    state_size = env.env.state_size
    action_space = env.env.action_space()
    brain_file = get_name_brain(args)
    brain = Brain(state_size, action_space, brain_file, args)
    all_agents = []
    for b_idx in range(args['agents_number']):
        all_agents.append(Agent(state_size, action_space, b_idx, brain_file, args))
    
    rewards_file = get_name_rewards(args)
    timesteps_file = get_name_timesteps(args)
    action_file = get_name_actions(args)
    states_file = get_name_states(args)
    rews_file = get_name_rews(args)
    
    rewards_path = './habitat_selection/rewards_files/'
    if not os.path.isdir(rewards_path):
        os.makedirs(rewards_path)
    brain_path = './habitat_selection/weights_files/'
    if not os.path.isdir(brain_path):
        os.makedirs(brain_path)
    timesteps_path = './habitat_selection/timesteps_files/'
    if not os.path.isdir(timesteps_path):
        os.makedirs(timesteps_path)
    actions_path = './habitat_selection/actions/'
    if not os.path.isdir(actions_path):
        os.makedirs(actions_path)
    states_path = './habitat_selection/states/'
    if not os.path.isdir(states_path):
        os.makedirs(states_path)
    rews_path = './habitat_selection/rewards_timesteps/'
    if not os.path.isdir(rews_path):
        os.makedirs(rews_path)
    loss_path = './habitat_selection/losses/'
    if not os.path.isdir(loss_path):
        os.makedirs(loss_path)
    
    env.run(all_agents, brain, rewards_file, timesteps_file, action_file, states_file, rews_file)