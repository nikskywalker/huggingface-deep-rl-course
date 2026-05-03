# Taxi-v3 Q-Learning Agent

This repository contains a Reinforcement Learning agent trained from scratch using the classical Q-Learning algorithm. The agent is trained to solve the `Taxi-v3` environment. This project is part of the Hugging Face Deep Reinforcement Learning Course (Unit 2).

## System Objective

The objective in the `Taxi-v3` environment is to navigate a 5x5 grid world, pick up a passenger at one of four specific locations, and drop them off at their requested destination. The environment consists of 500 discrete states and 6 discrete actions. The agent relies entirely on an updated Q-table (action-value function) to select the optimal path without using deep neural networks.

## Repository Structure

* `train_q_learning.py`: Core executable script containing the Q-table initialization, epsilon-greedy policy, training loop (Bellman equation updates), and evaluation pipeline.
* `requirements.txt`: Environment dependencies.

## Quick Start

### 1. Environment Setup

Isolate the dependencies using a virtual environment:

    cd 02_q_learning_taxi
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt

### 2. Execution

Run the main script to initiate training and evaluation:

    python train_q_learning.py

The script trains the Q-table over 1,000,000 episodes, evaluates it over 100 episodes using deterministic evaluation seeds, and saves the final Q-table as a numpy array (`qtable_taxi.npy`).

## Algorithm & Hyperparameters

The agent uses a tabular Q-Learning approach with an epsilon-greedy exploration strategy. 

* Total Episodes: 1,000,000
* Learning Rate: 0.7
* Gamma (Discount): 0.95
* Max Steps per Episode: 99
* Max Epsilon: 1.0
* Min Epsilon: 0.05
* Decay Rate: 0.005

## Performance Evaluation

Following the training phase, the agent's greedy policy was evaluated over 100 specific episodes. The model successfully passed the Hugging Face certification threshold.

* Validation Score (Mean - Std): 4.85