import gymnasium as gym
import numpy as np
import random
from tqdm import tqdm

def initialize_q_table(state_space, action_space):
    return np.zeros((state_space, action_space))

def greedy_policy(Qtable, state):
    # Exploitation: seleziona l'azione con il valore massimo
    return np.argmax(Qtable[state][:])

def epsilon_greedy_policy(Qtable, state, epsilon, env):
    random_num = random.uniform(0, 1)
    if random_num > epsilon:
        return greedy_policy(Qtable, state)
    else:
        # Exploration: azione casuale
        return env.action_space.sample()

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable, learning_rate, gamma):
    print("Inizio addestramento...")
    for episode in tqdm(range(n_training_episodes)):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        state, info = env.reset()
        
        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon, env)
            new_state, reward, terminated, truncated, info = env.step(action)
            
            # Aggiornamento di Q(s,a) tramite l'equazione di Bellman
            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )
            
            if terminated or truncated:
                break
            state = new_state
            
    return Qtable

def evaluate_agent(env, max_steps, n_eval_episodes, Q, eval_seed):
    print("Inizio valutazione...")
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if eval_seed:
            state, info = env.reset(seed=eval_seed[episode])
        else:
            state, info = env.reset()
            
        total_rewards_ep = 0
        
        for step in range(max_steps):
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward
            
            if terminated or truncated:
                break
            state = new_state
            
        episode_rewards.append(total_rewards_ep)
        
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward

def main():
    # Iperparametri definiti nel corso
    env_id = "Taxi-v3"
    n_training_episodes = 1_000_000
    n_eval_episodes = 100
    learning_rate = 0.7
    max_steps = 99
    gamma = 0.95
    max_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 0.005
    
    # Seed specifici richiesti dal corso per la valutazione
    eval_seed = [16,54,165,177,191,191,120,80,149,178,48,38,6,125,174,73,50,172,100,148,146,6,25,40,68,148,49,167,9,97,164,176,61,7,54,55,
                 161,131,184,51,170,12,120,113,95,126,51,98,36,135,54,82,45,95,89,59,95,124,9,113,58,85,51,134,121,169,105,21,30,11,50,65,12,43,82,145,152,97,106,55,31,85,38,
                 112,102,168,123,97,21,83,158,26,80,63,5,81,32,11,28,148]

    # Inizializzazione ambiente
    env = gym.make(env_id, render_mode="rgb_array")
    state_space = env.observation_space.n
    action_space = env.action_space.n
    
    # Inizializzazione Q-table
    Qtable_taxi = initialize_q_table(state_space, action_space)
    
    # Addestramento
    Qtable_taxi = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_taxi, learning_rate, gamma)
    
    # Valutazione
    mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_taxi, eval_seed)
    print(f"Risultato: Reward medio = {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Salvataggio Q-Table in locale
    np.save("qtable_taxi.npy", Qtable_taxi)
    print("Q-table salvata localmente come qtable_taxi.npy")

if __name__ == "__main__":
    main()