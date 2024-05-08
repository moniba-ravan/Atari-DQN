import argparse
import numpy as np
import gymnasium as gym
import torch
import config
import os
from utils import preprocess, append_to_csv, transform_action, show_time
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize
import matplotlib.pyplot as plt

import time 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=["ALE/Pong-v5", "CartPole-v1"], default="ALE/Pong-v5")
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'ALE/Pong-v5': config.PONG,
    'CartPole-v1': config.CartPole
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]
    if args.env == 'ALE/Pong-v5':
        env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30, scale_obs = True)
        env = gym.wrappers.FrameStack(env, env_config['obs_stack_size'])
    

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)

    # Load the saved model
    # saved_model_path = f'models/{args.env}_best.pt'
    # dqn.load_state_dict(torch.load(saved_model_path))

    # TODO: Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config).to(device)
    target_dqn.load_state_dict(dqn.state_dict()) # Copies weights from dqn
    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")
    mean_returns = []
    steps_done = 0

    start_training_time = time.time() # Start time measurement
    
    for episode in range(env_config['n_episodes']):

        start_episode_time = time.time()

        terminated = False
        truncated = False
        obs, info = env.reset()
        obs = torch.tensor(np.array(obs), device=device).float().unsqueeze(0)
        # print(f"obs from env.resest(): {obs}")
        
        while not terminated and not truncated:
            # TODO: Get action from DQN.
            action = dqn.act(obs).item()
            # print(f"action: {action}")

                
            next_obs, reward, terminated, truncated, info = env.step(transform_action(args.env, action))

            # Preprocess incoming observation.
            if not terminated and not truncated: 
                # next_obs = preprocess(np.array(next_obs), env=args.env).float().unsqueeze(0)
                next_obs = torch.tensor(np.array(next_obs), device=device).float().unsqueeze(0)
            else: # !!
                # Convert terminal state to torch
                next_obs = torch.tensor(np.array(next_obs), device=device).float().unsqueeze(0)
            
            # print(f"next_obs dim: {next_obs.size()}")
    
            
            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!

            action = torch.tensor(action, device=device)
            reward = torch.tensor(reward, device=device)
            terminated_torch = torch.tensor(terminated, device=device)
            # print(f"aciton: {action}")
            # print(f"reward: {reward}")
            # print(f"terminated: {terminated_torch}")
            
            memory.push(obs.squeeze(0),
                        action,
                        next_obs.squeeze(0),
                        reward,
                        terminated_torch,
                        )
            obs = next_obs
            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            if steps_done % env_config["train_frequency"] == 0:
                optimize(dqn,
                        target_dqn,
                        memory,
                        optimizer,
                        )
                
            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            if steps_done % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict()) # Copies weights from dqn

            steps_done += 1 # each action
        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            elapsed_time = time.time() - start_episode_time
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return} | Elapsed Time: {show_time(elapsed_time)}.')
            mean_returns.append(mean_return)

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return
                os.makedirs('models/ALE', exist_ok=True)
                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')
    
    total_training_time = time.time() - start_training_time
    print(f'Total Training Time: {show_time(total_training_time)}.')

    append_to_csv(env_config, mean_returns)
    #PLOT
    evaluation_episodes = [1 + args.evaluate_freq * index for index in range(len(mean_returns))]
    plt.plot(evaluation_episodes, mean_returns)
    plt.xlabel("Episode")
    plt.ylabel("Mean Return")
    plt.title("Evaluated mean returns")
    if args.env == 'ALE/Pong-v5':
        plt.savefig("PONG_mean_returns_plot.png")
        plt.savefig("PONG_mean_returns_plot.eps")
    else:
        plt.savefig("CARTPOLE_mean_returns_plot.png") 
        plt.savefig("CARTPOLE_mean_returns_plot.eps")   
    # Close environment after training is completed.
    env.close()
