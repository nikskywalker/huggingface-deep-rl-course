# Deep Reinforcement Learning Portfolio

This repository contains a collection of practical implementations and trained agents developed during the [Hugging Face Deep Reinforcement Learning Course](https://huggingface.co/deep-rl-course).

## Curriculum & Projects

* **[Unit 1: Proximal Policy Optimization (PPO)](./01_Unit_LunarLander_PPO)**
  * Environment: `LunarLander-v3`
  * Algorithm: PPO (Stable-Baselines3)
  * Objective: Controlled lunar module landing based on continuous physics variables.

* **[Unit 2: Q-Learning](./02_Unit_Taxi_QLearning)**
  * Environment: `Taxi-v3`
  * Algorithm: Tabular Q-Learning (Custom NumPy implementation)
  * Objective: Pathfinding and passenger logistics in a discrete grid world.

* **[Unit 3: Deep Q-Network (DQN)](./03_Unit_SpaceInvaders_DQN)**
  * Environment: `SpaceInvadersNoFrameskip-v4`
  * Algorithm: DQN (RL Baselines3 Zoo, CnnPolicy)
  * Objective: Vision-based reinforcement learning using raw pixel inputs and frame stacking.

## Repository Structure

Each module is designed to be entirely self-contained. Inside every unit's directory, you will find:
* The source code or configuration files required to reproduce the training phase.
* A `requirements.txt` file specifying the exact dependencies for that specific environment.
* A dedicated `README.md` detailing execution instructions, architecture, and validation metrics.