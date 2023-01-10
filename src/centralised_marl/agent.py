import numpy as np
from mind import Mind

class Agent:
    def __init__(self, id, loc, mind):
        self.id = id
        self.loc = loc
        self.current_state = None
        self.action = None
        self.next_state = None
        self.mind = mind
        self.input_size = mind.get_input_size()     
        self.output_size = mind.get_output_size()
        self.decision = None
        self.q_vals = None

    def update(self, reward, done):
        assert self.action != None, 'No Action'
        assert reward != None, 'No Reward'
        self.mind.remember([[[self.current_state]], [self.action], [[self.next_state]], [reward], [done]])

        #loss = self.mind.train()    #do I need this here or is it slowing down my code for no reason?

        self.action = None
        if not done:     #is this doing the same as the other if not done statement - should it be done == False
            self.current_state, self.next_state = self.next_state, None   
        else:
            self.current_state, self.next_state = None, None

    def get_losses(self):
        return self.mind.get_losses()
        pass

    def decide(self, state):
        self.action, self.q_vals = self.mind.decide(state)   #decision function choice
        return self.action, self.q_vals

    def get_state(self):
        return self.current_state

    def get_id(self):
        return self.id

    def get_loc(self):
        return self.loc

    def get_decision(self):
        assert self.decision != None, "Decision is requested without setting."
        return self.decision

    def set_decision(self, decision):
        self.decision = decision

    def clear_decision(self):
        self.decision = None

    def set_loc(self, loc):
        self.loc = loc

    def set_current_state(self, state):
        self.current_state = state

    def set_next_state(self, state):
        self.next_state = state

