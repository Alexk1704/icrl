import random
import numpy as np


class PrioritizedReplayBuffer:
    """ PER buffer that samples proportionately to the TD errors for each sample. """

    def __init__(self, buffer_size: int, input_dims, alpha: float = 0.6):
        """
        Args:
            buffer_size: max size.
            alpha: float, the strength of the prioritization
                (0.0 - no prioritization, 1.0 - full prioritization).
        """
        self.buffer_size = buffer_size
        self.counter = 0
        self.alpha = alpha

        self.state_memory = np.zeros((self.buffer_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.buffer_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.buffer_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.buffer_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.buffer_size, dtype=np.int32)
        self.priorities = np.ones(self.buffer_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.counter % self.buffer_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.priorities[index] = np.max(self.priorities)
        
        self.counter += 1

    def calculate_probabilities(self):
        """ Calculates probability of being sampled for each element in the buffer.
        Returns:
            probabilities: np.array, a list of values summing up to 1.0, with
                the probability of each element in the buffer being sampled
                according to its priority.
        """
        priorities = self.priorities ** self.alpha
        return priorities / sum(priorities)

    def calculate_importance(self, probs):
        """ Calculates the importance sampling bias correction. """
        importance = 1.0 / self.buffer_size * 1.0 / probs
        return importance / np.max(importance)

    def sample_buffer(self, batch_size):
        probs = self.calculate_probabilities()
        max_mem = min(self.counter, self.buffer_size)
        batch_indices = np.random.choice(max_mem, batch_size, p=probs)

        importance = self.calculate_importance(probs[batch_indices])

        states = self.state_memory[batch_indices]
        states_ = self.new_state_memory[batch_indices]
        rewards = self.reward_memory[batch_indices]
        actions = self.action_memory[batch_indices]
        terminal = self.terminal_memory[batch_indices]        

        return states, actions, rewards, states_, terminal, importance

    def update_priorities(self, indices, td_errors, offset=0.01):
        """Updates the priorities for a batch given the TD errors.
        Args:
            indices: np.array, the list of indices of the priority list to be
                updated.
            td_errors: np.array, the list of TD errors for those indices. The
                priorities will be updated to the TD errors plus the offset.
            offset: float, small positive value to ensure that all trajectories
                have some probability of being selected.
        """
        for index, error in zip(indices, td_errors):
            self.priorities[index] = abs(error) + offset