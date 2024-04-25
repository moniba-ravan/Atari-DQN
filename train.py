import argparse

import gymnasium as gym
import torch

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v1'], default='CartPole-v1')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v1': config.CartPole
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

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

    steps_done = 0 # !!
    for episode in range(env_config['n_episodes']):
        terminated = False
        obs, info = env.reset()

        obs = preprocess(obs, env=args.env).unsqueeze(0)

        
        while not terminated:
            # TODO: Get action from DQN.
            action = dqn.act(obs).item() # !!
            # print("action:\n", action.shape,"obs:\n", obs.shape)
            # Act in the true environment.
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Preprocess incoming observation.
            if not terminated: 
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
                # print("not: ", next_obs)
            else:
                next_obs = torch.tensor(next_obs, device=device).float().unsqueeze(0)
                # print("terminal: ", next_obs)
            
            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
            
            # !!
            action = torch.tensor(action, device=device)
            reward = torch.tensor(reward, device=device).float()
            terminated_torch = torch.tensor(terminated, device=device)
            
            
            memory.push(obs,
                        action,
                        next_obs,
                        reward,
                        terminated_torch,
                        )
            obs = next_obs
            # !!

            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.

            # !!
            if steps_done % env_config["train_frequency"] == 0:
                optimize(dqn,
                        target_dqn,
                        memory,
                        optimizer,
                        )
                dqn.update_eps_threshold()
            # !!

            # TODO: Update the target network every env_config["target_update_frequency"] steps.

            # !!
            if steps_done % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict()) # Copies weights from dqn

            steps_done += 1 # each action
            # !!
            # end of while loop


        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                # !! create model folder
                directory = "models"
                if not os.path.exists(directory):
                    # Create the directory
                    os.makedirs(directory)
                    print(f"Directory '{directory}' created successfully.")
                else:
                    print(f"Directory '{directory}' already exists.")

                # !!
                torch.save(dqn, f'{directory}/{args.env}_best.pt')
        
    # Close environment after training is completed.
    env.close()
