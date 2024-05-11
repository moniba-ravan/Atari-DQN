import argparse

import gymnasium as gym
import torch
import numpy as np
import config
from utils import transform_action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=["ALE/Pong-v5", "CartPole-v1"], default="ALE/Pong-v5")
parser.add_argument('--path', type=str, help='Path to stored DQN model.')
parser.add_argument('--n_eval_episodes', type=int, default=1, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--render', dest='render', action='store_true', help='Render the environment.')
parser.add_argument('--save_video', dest='save_video', action='store_true', help='Save the episodes as video.')
parser.set_defaults(render=False)
parser.set_defaults(save_video=True)

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'ALE/Pong-v5': config.PONG,
    'CartPole-v1': config.CartPole,
}


def evaluate_policy(dqn, env, env_config, args, n_episodes, render=False, verbose=False):
    """Runs {n_episodes} episodes to evaluate current policy."""
    total_return = 0
    for i in range(n_episodes):
        obs, info = env.reset()
        obs = torch.tensor(np.array(obs), device=device).float().unsqueeze(0)
       
        terminated = False
        truncated = False

        episode_return = 0

        while not terminated and not truncated:
            if render:
                env.render()
            
            action = dqn.act(obs, exploit=True).item()
            
            obs, reward, terminated, truncated, info = env.step(transform_action(args.env, action))
            obs = torch.tensor(np.array(obs), device=device).float().unsqueeze(0)
            episode_return += reward
        
        total_return += episode_return
        
        if verbose:
            print(f'Finished episode {i+1} with a total return of {episode_return}')

    
    return total_return / n_episodes

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

    if args.save_video:
        env = gym.make(args.env, render_mode='rgb_array')
        video_path = "./video/"
        if args.env == "ALE/Pong-v5":
            video_path += "pong/"
        else:
            video_path += "cartpole/"
        env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=lambda episode_id: True)

    if args.env == 'ALE/Pong-v5':
        # Setup the size of each frame and number of frames.
        env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30, scale_obs = True)
        env = gym.wrappers.FrameStack(env, env_config['obs_stack_size'])
        
    # Load model from provided path.
    dqn = torch.load(args.path, map_location=torch.device(device))
    dqn.eval()

    mean_return = evaluate_policy(dqn, env, env_config, args, args.n_eval_episodes, render=args.render and not args.save_video, verbose=True)
    print(f'The policy got a mean return of {mean_return} over {args.n_eval_episodes} episodes.')

    env.close()