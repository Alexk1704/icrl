import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, input_dims):
        self.buffer_size    = buffer_size
        self.counter        = 0

        self.state_memory       = np.zeros((self.buffer_size, *input_dims),dtype=np.float32)
        self.new_state_memory   = np.zeros((self.buffer_size, *input_dims), dtype=np.float32)
        self.action_memory      = np.zeros(self.buffer_size,dtype=np.int32)
        self.reward_memory      = np.zeros(self.buffer_size,dtype=np.float32)
        self.terminal_memory    = np.zeros(self.buffer_size,dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.counter % self.buffer_size
        
        self.state_memory[index]        = state
        self.new_state_memory[index]    = state_
        self.reward_memory[index]       = reward
        self.action_memory[index]       = action
        self.terminal_memory[index]     = 1 - int(done)
    
        self.counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.counter, self.buffer_size)
        batch_indices = np.random.choice(max_mem, batch_size, replace=False)

        states      = self.state_memory[batch_indices]
        states_     = self.new_state_memory[batch_indices]
        rewards     = self.reward_memory[batch_indices]
        actions     = self.action_memory[batch_indices]
        terminal    = self.terminal_memory[batch_indices]

        return states, actions, rewards, states_, terminal, batch_indices, None
    

class PrioritizedReplayBuffer:
    """ PER buffer that samples proportionately to the TD errors for each sample. """

    def __init__(self, buffer_size, input_dims, per_alpha=0.6, per_beta=1.0, per_eps=1e-6):
        """
        Args:
            buffer_size: max buffer size.
            input_dims: observation dimension.
            per_alpha: the strength of the prioritization (0.0 - no prioritization, 1.0 - full prioritization).
            per_beta: beta controls how much prioritization to apply, should start small (0,4-0,6 and anneal to 1).
            per_eps: small constant ensuring that each sample has some non-zero probability of being drawn.
        """
        self.buffer_size    = buffer_size
        self.counter        = 0
        self.alpha          = per_alpha
        self.beta           = per_beta
        self.eps            = per_eps

        self.state_memory       = np.zeros((self.buffer_size, *input_dims), dtype=np.float32)
        self.new_state_memory   = np.zeros((self.buffer_size, *input_dims), dtype=np.float32)
        self.action_memory      = np.zeros(self.buffer_size, dtype=np.int32)
        self.reward_memory      = np.zeros(self.buffer_size, dtype=np.float32)
        self.terminal_memory    = np.zeros(self.buffer_size, dtype=np.int32)
        self.priorities         = np.ones(self.buffer_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.counter % self.buffer_size

        self.state_memory[index]        = state
        self.new_state_memory[index]    = state_
        self.reward_memory[index]       = reward
        self.action_memory[index]       = action
        self.terminal_memory[index]     = 1 - int(done)
        self.priorities[index]          = np.max(self.priorities)  # assign a priority, initially set to a high value.
        
        self.counter += 1

    def calculate_probabilities(self):
        """ Calculates probability of being sampled for each element in the buffer.
        Returns:
            probabilities: 
                returns a probability distribution P(i) of how likely it is,
                that a buffer element will be retrieved according to its priority.
                We use the proportional variant here (see DeepMind paper).
        """
        priorities = self.priorities[:self.counter] ** self.alpha
        return priorities / sum(priorities[:self.counter])

    def calculate_importance(self, probs):
        """ Calculates the importance sampling bias correction. """
        importance = ((1.0 / self.counter) * (1.0 / probs))**self.beta  
        return importance / np.max(importance)  # max w_i = 1 for stability

    def sample_buffer(self, batch_size):
        """ Sample based on priorities, experiences with a higher priority are more likely to be sampled. """
        # calculate probability distribution based on the importance (so samples with no/low importance still have a small chance to be sampled)
        probs = self.calculate_probabilities()
        # generate a probability distribution of size N (mini-batch size) based on the importance of already stored samples.
        possible_indices = np.arange(0, min(self.counter, self.buffer_size))
        # draw from distribution (samples with higher importance/td-error are drawn more frequently)
        batch_indices = np.random.choice(possible_indices, batch_size, p=probs[:self.counter])
        # recalculate the importance of each sample dran
        importance = self.calculate_importance(probs[batch_indices])

        states      = self.state_memory[batch_indices]
        states_     = self.new_state_memory[batch_indices]
        rewards     = self.reward_memory[batch_indices]
        actions     = self.action_memory[batch_indices]
        terminal    = self.terminal_memory[batch_indices]        

        return states, actions, rewards, states_, terminal, batch_indices, importance

    def update_priorities(self, indices, td_errors):
        """Updates the priorities for a batch given the TD errors (pred. Q-values minus target Q-values).
        Experiences that are sampled often are given low priority values to balance sampling.
        Args:
            indices: np.array, the list of indices of the priority list to be updated.
            td_errors: np.array, the list of TD errors for those indices. The
                priorities will be updated to the TD errors plus the offset.
            eps: float, small positive value to ensure that all trajectories
                have some probability of being selected.
        """
        for index, error in zip(indices, td_errors):
            self.priorities[index] = abs(error) + self.eps