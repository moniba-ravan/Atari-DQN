# Deep Q-Network (DQN) for Reinforcement Learning

## Overview
This project implements a Deep Q-Network (DQN) to solve reinforcement learning tasks in two different environments: CartPole-v1 and Pong (using ALE/Pong-v5 environment).

## Files

### `config.py`
This file contains configurations and hyperparameters for the DQN algorithm. It allows users to easily adjust parameters such as memory size, number of episodes, batch size, learning rate, and exploration rate, among others.

### `dqn.py`
Defines the structure of the Deep Q-Network (DQN) model using PyTorch. It includes classes for both the DQN model and the replay memory buffer.

### `train.py`
This script contains the training loop for the DQN algorithm. It initializes the environment, trains the DQN model, and periodically evaluates its performance.

### `evaluate.py`
Provides functionality to evaluate the trained DQN model on the specified environment. It loads a pre-trained model and runs evaluation episodes to measure its performance.

## Usage

1. **Setting Hyperparameters**: Modify the hyperparameters in `config.py` according to your requirements and the environment you want to train the DQN for.

2. **Training the Model**: Run `train.py` to train the DQN model. The script will iterate over episodes, updating the model parameters and periodically evaluating the model's performance.

3. **Evaluating the Model**: Use `evaluate.py` to assess the performance of the trained model. You can specify the path to the saved model and the number of evaluation episodes to run.

## Dependencies

- Python 3.x
- PyTorch
- Gymnasium
- NumPy
- Matplotlib

## Additional Notes

- Make sure to have the necessary dependencies installed before running the scripts.
- The project includes environments for both CartPole-v1 and Pong. You can choose the desired environment using command-line arguments when running the scripts.

Feel free to explore and customize the code to adapt it to different environments or experiment with various hyperparameters for reinforcement learning tasks.