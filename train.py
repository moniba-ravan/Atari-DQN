import argparse
import numpy as np
import gymnasium as gym
import torch
import config
import os
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=["ALE/Pong-v5", 'CartPole-v1'], default="ALE/Pong-v5")
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
    # TODO: Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config).to(device)
    target_dqn.load_state_dict(dqn.state_dict()) # Copies weights from dqn
    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")
    steps_done = 0
    for episode in range(env_config['n_episodes']):
        terminated = False
        truncated = False
        obs, info = env.reset()
        obs = torch.tensor(np.array(obs), device=device).float().unsqueeze(0)
        # print(f"obs from env.resest(): {obs}")
        
        while not terminated and not truncated:
            # TODO: Get action from DQN.
            action = dqn.act(obs).item()
            # print(f"action: {action}")

            if args.env == 'ALE/Pong-v5':
                # Act in the true environment.
                # 2, 3
                if action == 0:
                    transformed_action = torch.tensor([2], device=device) # UP
                else:
                    transformed_action = torch.tensor([3], device=device) # DOWN
            else:
                transformed_action = action
                
            next_obs, reward, terminated, truncated, info = env.step(transformed_action)

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
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return
                os.makedirs('models/ALE', exist_ok=True)
                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')
        
    # Close environment after training is completed.
    env.close()
