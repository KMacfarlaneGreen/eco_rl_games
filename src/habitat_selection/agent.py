import numpy as np 
from uniform_experience_replay import Memory as UER 

MAX_EPSILON = 1.0    #decay rate
MIN_EPSILON = 0.01

class Agent(object):
    
    epsilon = MAX_EPSILON
    #beta = MIN_BETA

    def __init__(self, state_size, action_size, bee_index, brain, arguments):
        self.state_size = state_size
        self.action_size = action_size
        self.bee_index = bee_index
        self.brain = brain #Brain(self.state_size, self.action_size, brain_name, arguments)    

        self.memory_model = arguments['memory']

        if self.memory_model == 'UER':     #move memory to brain
            self.memory = UER(arguments['memory_capacity'])

        else:
            print('Invalid memory model!')

        #where should these be?
        self.update_target_frequency = arguments['target_frequency']
        self.max_exploration_step = arguments['maximum_exploration']
        self.step = 0

        self.test = arguments['test']
        if self.test:
            self.epsilon = MIN_EPSILON

    def greedy_actor(self, state):   #get agent action
        if np.random.rand() <= self.epsilon:
            return np.random.randrange(self.action_size)
        else:
            return np.argmax(self.brain.predict_one_sample(state))


    def find_targets_uer(self, batch):  
        batch_len = len(batch)

        states = np.array([o[0] for o in batch])
        states_ = np.array([o[3] for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)
        pTarget_ = self.brain.predict(states_, target=True)

        x = np.zeros((batch_len, self.state_size))
        y = np.zeros((batch_len, self.action_size))
        errors = np.zeros(batch_len)

        for i in range(batch_len):
            o = batch[i]
            s = o[0]
            a = o[1][self.bee_index]  
            r = o[2][self.bee_index]  #bee_index corresponds to agent
            s_ = o[3]
            done = o[4]

            t = p[i]
            old_value = t[a]
            if done:
                t[a] = r
            else:
                t[a] = r + self.gamma * np.amax(pTarget_[i])

            x[i] = s
            y[i] = t
            errors[i] = np.abs(t[a] - old_value)

        return [x, y]

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

        else:
            print('Invalid memory model!')
    
    def replay(self): 

        if self.memory_model == 'UER':
            batch = self.memory.sample(self.batch_size)
            x, y = self.find_targets_uer(batch)
            loss = self.brain.opt(x, y)
            return loss.detach().numpy()

        else:
            print('Invalid memory model!')

    def update_target_model(self):   
        if self.step % self.update_target_frequency == 0:
            self.brain.update_target_model()

    def get_index(self):   
        return self.bee_index