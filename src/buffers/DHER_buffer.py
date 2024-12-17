import random
import numpy as np
import torch

class DHERReplayBuffer:
    def __init__(self, buffer_size, batch_size, get_q_value_func, gamma, alpha=0.6):
        """
        Initialize a Dynamic Hindsight Experience Replay Buffer.

        Args:
            buffer_size (int): Maximum number of transitions to store in the buffer.
            batch_size (int): Number of transitions to sample from the buffer.
            get_q_value_func (callable): Function to compute Q-values for states and actions.
            gamma (float): Discount factor for future rewards.
            alpha (float): Determines the extent of prioritization based on TD Errors.
        """
        self.buffer = []
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.get_q_value = get_q_value_func
        self.gamma = gamma
        self.alpha = alpha
        self.position = 0
        self.name = "DHERReplayBuffer"

    def add(self, state, action, reward, next_state, done, goal=None):
        if goal is None:
            goal = next_state
        transition = (state, action, reward, next_state, done, goal)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.buffer_size

    def add_episode(self, episode_transitions):
        for transition in episode_transitions:
            self.add(*transition)
        her_transitions = self._dynamic_goal_selection(episode_transitions)
        for her_transition in her_transitions:
            self.add(*her_transition)

    def _dynamic_goal_selection(self, episode_transitions):
        td_errors = []
        for state, action, reward, next_state, done in episode_transitions:
            current_q_value = self.get_q_value(state, action)
            next_q_value = np.max(self.get_q_value(next_state)) if not done else 0
            td_error = np.abs(reward + self.gamma * next_q_value - current_q_value)
            td_errors.append(td_error)
        
        probabilities = np.power(td_errors, self.alpha)
        probabilities /= np.sum(probabilities)
        selected_indices = np.random.choice(len(episode_transitions), size=3, replace=False, p=probabilities)

        her_transitions = []
        for idx in selected_indices:
            state, action, _, next_state, done = episode_transitions[idx]
            alternative_goal = next_state
            new_reward = float(np.all(np.isclose(next_state, alternative_goal)))
            her_transitions.append((state, action, new_reward, next_state, done, alternative_goal))

        return her_transitions

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones, goals = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.int64),  # Ensure actions are of integer type
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.bool),
            torch.tensor(np.array(goals), dtype=torch.float32)
        )
    def __len__(self):
        return len(self.buffer)
