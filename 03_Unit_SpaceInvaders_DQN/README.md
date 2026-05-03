# Space Invaders DQN Agent

This repository contains the configuration to train a Deep Q-Network (DQN) agent on the `SpaceInvadersNoFrameskip-v4` environment. This project is part of the Hugging Face Deep Reinforcement Learning Course (Unit 3) and utilizes the `RL Baselines3 Zoo` framework.

## System Objective

The objective is to train an agent to play Atari Space Invaders directly from pixel inputs. The environment is wrapped using standard Atari wrappers (grayscale, frame stacking). The model employs a Convolutional Neural Network (`CnnPolicy`) to process the visual state and estimate Q-values for the available actions.

## Repository Structure

* `dqn.yml`: The hyperparameter configuration file defining the environment constraints, memory buffer, learning rate, and exploration schedules.
* `requirements.txt`: Environment dependencies required to run the Zoo framework and the Atari ROMs.

## Quick Start

### 1. Environment Setup

Isolate the dependencies using a virtual environment:

    cd 03_Unit_SpaceInvaders_DQN
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    
    # System dependencies for Atari (Linux/WSL only)
    # sudo apt-get install swig cmake ffmpeg
    
    pip install -r requirements.txt

### 2. Execution (Training)

Unlike previous implementations, this agent is trained entirely via the RL Zoo command-line interface. Run the following command to initiate training for 1,000,000 timesteps using the provided configuration:

    python -m rl_zoo3.train --algo dqn --env SpaceInvadersNoFrameskip-v4 -f logs/ -c dqn.yml

*Note: Training an agent on pixel data requires significant computational power. Execution on a GPU is highly recommended.*

### 3. Evaluation

To evaluate the agent's performance after training without rendering the environment:

    python -m rl_zoo3.enjoy --algo dqn --env SpaceInvadersNoFrameskip-v4 --no-render --n-timesteps 5000 --folder logs/

## Algorithm & Hyperparameters

The agent utilizes a standard Deep Q-Network implementation via Stable-Baselines3. Key hyperparameters defined in `dqn.yml` include:

* Total Timesteps: 1,000,000
* Replay Buffer Size: 100,000
* Batch Size: 32
* Learning Rate: 0.0001
* Policy: CnnPolicy
* Frame Stacking: 4 frames

## Performance Evaluation

Following the training phase, the agent was evaluated on the Hugging Face platform to ensure the base criteria for the Space Invaders environment were met.

* Validation Score (Mean - Std): 505.43
