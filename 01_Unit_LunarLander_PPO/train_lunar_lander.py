import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from huggingface_sb3 import package_to_hub

def main():
    # Define the environment identifier
    env_id = "LunarLander-v3"
    
    # ==========================================
    # 1. Environment Setup
    # ==========================================
    # We create a vectorized environment stacking 16 independent instances.
    # Vectorization allows the agent to collect diverse experiences simultaneously,
    # which stabilizes the training process and significantly speeds it up.
    env = make_vec_env(env_id, n_envs=16)

    # ==========================================
    # 2. Model Architecture and Configuration
    # ==========================================
    # We use Proximal Policy Optimization (PPO), an actor-critic algorithm.
    # MlpPolicy (Multi-Layer Perceptron) is chosen because the observation space 
    # is a 1D vector (coordinates, velocities, etc.), not a 2D image.
    model = PPO(
        policy='MlpPolicy',
        env=env,
        n_steps=1024,      # Number of steps to run for each environment per update
        batch_size=64,     # Minibatch size for the optimization step
        n_epochs=4,        # Number of epochs when optimizing the surrogate loss
        gamma=0.999,       # Discount factor: close to 1 means future rewards are heavily weighted
        gae_lambda=0.98,   # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        ent_coef=0.01,     # Entropy coefficient: encourages exploration by penalizing deterministic actions
        verbose=1          # Set to 1 to print training metrics to the console
    )

    # ==========================================
    # 3. Training Phase
    # ==========================================
    print("Starting training phase...")
    # Train the agent for 1,000,000 timesteps. This duration is generally 
    # sufficient for the LunarLander environment to converge to a stable policy.
    model.learn(total_timesteps=1_000_000)
    
    # Save the trained model locally
    model_name = "ppo-LunarLander-v3"
    model.save(model_name)
    print(f"Model successfully saved locally as {model_name}.zip")

    # ==========================================
    # 4. Evaluation Phase
    # ==========================================
    print("Starting evaluation phase...")
    # Create a separate, single environment specifically for evaluation.
    # Wrapping it in a Monitor is required by SB3 to properly log episode statistics.
    eval_env = Monitor(gym.make(env_id, render_mode='rgb_array'))
    
    # Evaluate the policy over 10 episodes.
    # deterministic=True forces the agent to always select the action with the highest probability.
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Evaluation Results: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    # ==========================================
    # 5. Hugging Face Hub Integration (Optional)
    # ==========================================
    # Uncomment and configure the following block to push the trained model, 
    # metrics, and a gameplay video to the Hugging Face Hub.
    
    # repo_id = "YourUsername/ppo-LunarLander-v3"
    # env_eval_hub = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
    
    # package_to_hub(
    #     model=model,
    #     model_name=model_name,
    #     model_architecture="PPO",
    #     env_id=env_id,
    #     eval_env=env_eval_hub,
    #     repo_id=repo_id,
    #     commit_message="Upload trained PPO agent for LunarLander-v3"
    # )

if __name__ == "__main__":
    main()