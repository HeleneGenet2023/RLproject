import random
import torch
import numpy as np

class HERReplayBuffer:
    def __init__(self, buffer_size, batch_size, goal_selection_strategy='final'):
        self.buffer = []
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.position = 0
        self.goal_selection_strategy = goal_selection_strategy
        self.name = "HERReplayBuffer"
    
    def add(self, state, action, reward, next_state, done, goal=None):
        """
        Add a transition to the buffer with optional goal
        
        If no goal is provided, use the next_state as a default goal
        """
        if goal is None:
            goal = next_state
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done, goal))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done, goal)
        
        self.position = (self.position + 1) % self.buffer_size
    
    def add_episode(self, episode_transitions):
        """
        Add an entire episode to the buffer with optional HER
        """
        # Add original transitions
        for transition in episode_transitions:
            self.add(*transition)
        
        # Optionally generate and add HER transitions if goal_selection_strategy is set
        if self.goal_selection_strategy:
            her_transitions = self._generate_her_goals(episode_transitions)
            for her_transition in her_transitions:
                self.add(*her_transition)
    
    def _generate_her_goals(self, episode_transitions):
        her_transitions = []
        
        for i, transition in enumerate(episode_transitions):
            state, action, reward, next_state, done, goal = transition
            
            # Select alternative goal based on strategy
            if self.goal_selection_strategy == 'final':
                alternative_goal = episode_transitions[-1][3]  # final state
            elif self.goal_selection_strategy == 'random':
                alternative_goal = random.choice(episode_transitions)[3]
            else:
                raise ValueError(f"Unknown goal selection strategy: {self.goal_selection_strategy}")
            
            # Recalculate reward with the alternative goal
            alternative_reward = float(np.all(np.isclose(next_state, alternative_goal)))
            
            her_transitions.append((
                state, action, alternative_reward, 
                next_state, done, alternative_goal
            ))
        
        return her_transitions
    
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones, goals = zip(*batch)
        
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(goals, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)