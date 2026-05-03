# LunarLander-v3 PPO Agent

This repository provides a Deep Reinforcement Learning implementation designed to solve the `LunarLander-v3` environment. The agent is trained using the Proximal Policy Optimization (PPO) algorithm. This work was developed following the Hugging Face Deep Reinforcement Learning Course curriculum.

## System Objective

The agent's objective is to execute a controlled landing of a lunar module on a designated target area. The policy controls three discrete engines (main, left, right) to manage velocity, spatial positioning, and angular orientation. The environment yields positive rewards for safe, zero-velocity landings and issues penalties for collisions or excessive fuel consumption. The environment is considered solved when the agent achieves a running average score of 200 or higher.

## Repository Structure

* `train_lunar_lander.py`: Core executable script containing the vectorized environment initialization, PPO model architecture, training loop, and evaluation pipeline.
* `requirements.txt`: Exact environment dependencies required to execute the code.

## Quick Start

### 1. Environment Setup

It is recommended to isolate the dependencies using a virtual environment:

    git clone <your-repository-url>
    cd <your-repository-folder>/01_lunar_lander_ppo
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt

### 2. Execution

Run the main script to initiate the training and evaluation phases:

    python train_lunar_lander.py

The pipeline automatically handles vectorization (16 parallel environments), trains the model for 1,000,000 timesteps, saves the output binary as `ppo-LunarLander-v3.zip`, and runs a 10-episode deterministic evaluation.

## Model Architecture & Hyperparameters

The agent utilizes an `MlpPolicy` architecture, optimized for the 1D observation space of the environment. The training hyperparameters are configured as follows:

* Total Timesteps: 1,000,000
* n_steps: 1024
* batch_size: 64
* n_epochs: 4
* gamma: 0.999
* gae_lambda: 0.98
* ent_coef: 0.01

## Performance Evaluation

Following the training phase, the model's deterministic policy was evaluated over 10 consecutive episodes.

* Mean Reward: 269.83
* Standard Deviation: 17.01

