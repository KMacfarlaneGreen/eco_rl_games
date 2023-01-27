import numpy as np 
from uniform_experience_replay import Memory as UER 
from prioritised_experience_replay import Memory as PER

MAX_EPSILON = 1.0    #decay rate
MIN_EPSILON = 0.01

MIN_BETA = 0.4
MAX_BETA = 1.0

class Agent(object):
    
    epsilon = MAX_EPSILON
    beta = MIN_BETA

    def __init__(self, state_size, action_size, bee_index, brain, arguments):
        self.state_size = state_size
        self.action_size = action_size
        self.bee_index = bee_index
        self.brain = brain #Brain(self.state_size, self.action_size, brain_name, arguments)    
        self.gamma = 0.95
        self.memory_model = arguments['memory']

        if self.memory_model == 'UER':     #move memory to brain
            self.memory = UER(arguments['memory_capacity'])

        elif self.memory_model == 'PER':
            self.memory = PER(arguments['memory_capacity'], arguments['prioritization_scale'])

        else:
            print('Invalid memory model!')

        #where should these be?
        self.update_target_frequency = arguments['target_frequency']
        self.max_exploration_step = arguments['maximum_exploration']
        self.batch_size = arguments['batch_size']
        self.step = 0

        self.test = arguments['test']
        if self.test:
            self.epsilon = MIN_EPSILON

    def greedy_actor(self, state):   #get agent action
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.brain.predict_one_sample(state))


    def find_targets_uer(self, batch):  
        batch_len = len(batch)

        states = np.array([o[0] for o in batch])
        states_ = np.array([o[3] for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)
        pTarget_ = self.brain.predict(states_, target=True)

        x = np.zeros((batch_len))
        y = np.zeros((batch_len))
        errors = np.zeros(batch_len)

        for i in range(batch_len):
            o = batch[i] 
            s = o[0]       #state
            a = o[1][self.bee_index]      #action corresponding to an agent index
            r = o[2][self.bee_index]       #reward corresponding to same agent index
            s_ = o[3]                     #next state
            done = o[4]                   #done

            t = p[i]                      #q values for state s
            old_value = t[a]              #q value for action taken from state s
            if done:
                t[a] = r                   #if done, q value is just reward
            else:
                t[a] = r + self.gamma * np.amax(pTarget_[i]) #if not done, q value is reward + discounted max q value of next state

            x[i] = old_value
            y[i] = t[a]
            errors[i] = np.abs(t[a] - old_value)

        return [x, y]

    def find_targets_per(self, batch):
        batch_len = len(batch)

        states = np.array([o[1][0] for o in batch])
        states_ = np.array([o[1][3] for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)
        pTarget_ = self.brain.predict(states_, target=True)

        x = np.zeros((batch_len))
        y = np.zeros((batch_len))
        errors = np.zeros(batch_len)

        for i in range(batch_len):
            o = batch[i] 
            s = o[1][0]       #state
            a = o[1][1][self.bee_index]      #action corresponding to an agent index
            r = o[1][2][self.bee_index]       #reward corresponding to same agent index
            s_ = o[1][3]                     #next state
            done = o[1][4]                   #done
                 #done

            t = p[i]                      #q values for state s
            old_value = t[a]              #q value for action taken from state s
            if done:
                t[a] = r                   #if done, q value is just reward
            else:
                t[a] = r + self.gamma * np.amax(pTarget_[i]) #if not done, q value is reward + discounted max q value of next state

            x[i] = old_value
            y[i] = t[a]
            errors[i] = np.abs(t[a] - old_value)

        return [x, y, errors]

    def decay_epsilon(self): 
        # slowly decrease Epsilon based on our experience
        self.step += 1

        if self.test:
            self.epsilon = MIN_EPSILON
        else:
            if self.step < self.max_exploration_step:
                self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * (self.max_exploration_step - self.step)/self.max_exploration_step
            else:
                self.epsilon = MIN_EPSILON

    def observe(self, sample):

        if self.memory_model == 'UER':
            self.memory.remember(sample)

        elif self.memory_model == 'PER':
            _, _, errors = self.find_targets_per([[0, sample]])
            self.memory.remember(sample, errors[0])

        else:
            print('Invalid memory model!')
    
    def replay(self): 

        if self.memory_model == 'UER':
            batch = self.memory.sample(self.batch_size)
            x, y = self.find_targets_uer(batch)
            loss = self.brain.opt(x, y)
            return loss.detach().numpy()

        elif self.memory_model == 'PER':
            [batch, batch_indices, batch_priorities] = self.memory.sample(self.batch_size)
            x, y, errors = self.find_targets_per(batch)

            normalized_batch_priorities = [float(i) / sum(batch_priorities) for i in batch_priorities]
            importance_sampling_weights = [(self.batch_size * i) ** (-1 * self.beta)
                                           for i in normalized_batch_priorities]
            normalized_importance_sampling_weights = [float(i) / max(importance_sampling_weights)
                                                      for i in importance_sampling_weights]
            sample_weights = [errors[i] * normalized_importance_sampling_weights[i] for i in range(len(errors))]

            self.brain.opt(x, y, np.array(sample_weights))

            self.memory.update(batch_indices, errors)

        else:
            print('Invalid memory model!')

    def update_target_model(self):   
        if self.step % self.update_target_frequency == 0:
            self.brain.update_target_model()

    def get_index(self):   
        return self.bee_index