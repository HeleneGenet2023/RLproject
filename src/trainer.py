import numpy as np

def train(env, agent, n_episodes, max_t, epsilon_start, epsilon_end, epsilon_decay):
    scores = []
    training_loss = []
    epsilon = epsilon_start
    
    for episode in range(1, n_episodes + 1):
        state = env.reset()  # In 0.25.0, reset() returns just the state
        total_reward = 0
        episode_loss = 0.0  # Initialize as float
        
        for t in range(max_t):
            action = agent.select_action(state, epsilon)
            
            # Unpack 4 values from step method
            next_state, reward, done, _ = env.step(action)
            
            # Ensure step returns a numeric loss
            current_loss = agent.step(state, action, reward, next_state, done)
            episode_loss += current_loss if current_loss is not None else 0.0
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Avoid division by zero
        training_loss.append(episode_loss / max(1, t))
        scores.append(total_reward)
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores[-100:]):.2f}")
    
    return scores, training_loss