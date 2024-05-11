import torch
import csv
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform_action(env, action):
    """
    Transform actions based on the environment in gym.

    Args:
        env (str): Environment name.
        action (int): Action index.

    Returns:
        torch.Tensor: Transformed action.
    """
    if env == 'ALE/Pong-v5':
            # Act in the true environment.
            if action == 2:
                return torch.tensor([0], device=device) # NOPT
            elif action == 0:
                return torch.tensor([2], device=device) # UP
            elif action == 1:
                return  torch.tensor([3], device=device) # DOWN
    else:
        return action

def show_time(total_seconds):
    """
    Convert total seconds into hours, minutes, and seconds format.

    Args:
        total_seconds (float): Total seconds.

    Returns:
        str: Time in hours, minutes, and seconds format.
    """
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours} hours, {minutes} minutes, {seconds} seconds"

def append_to_csv(config, mean_returns, env='ALE/Pong-v5'):
    """
    Append mean returns to a CSV file.

    Args:
        config (dict): Configuration parameters.
        mean_returns (list): List of mean returns.
        env (str): CSV file path (default: 'mean_return.csv').
    """
    if env == 'ALE/Pong-v5':
        csv_file = 'csv/pong_mean_return.csv'
    else:
        csv_file = 'csv/cartpole_mean_return.csv'
    os.makedirs("csv", exist_ok=True)
    if not os.path.exists(csv_file):
        # If the file doesn't exist, create it and open in write mode
        with open(csv_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(config.keys())+['Mean Return'])
            writer.writeheader()
    mean_returns_str = ','.join(map(str, mean_returns))
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list(config.values()) + [mean_returns_str])

def read_csv(env):
    if env == 'ALE/Pong-v5':
        csv_file = 'csv/pong_mean_return.csv'
    else:
        csv_file = 'csv/cartpole_mean_return.csv'
    # Read from the CSV file
    with open(csv_file, 'r', newline='') as file:
        list_of_trains = list()
        list_of_configs = list()
        reader = csv.reader(file)
        column_names = next(reader)
        for row in reader:
            config = {}
            mean_returns = list()
            for idx, value in enumerate(row):
                if idx == 0:
                    continue
                elif idx == 1:
                    config[column_names[idx]] = value
                elif value.isdigit():
                    config[column_names[idx]] = int(value)
                else:
                    try:
                        config[column_names[idx]] = float(value)
                    except ValueError:
                        mean_returns = list(map(float, value.split(',')))
            # print(config)
            # print(mean_returns)
            list_of_trains.append(mean_returns)
            list_of_configs.append(config)
        return list_of_configs, list_of_trains

def save_plot(mean_returns, evaluate_freq, env='ALE/Pong-v5'):
    """
    Draw and save plot of mean returns of current model.

    Args:
        mean_returns (list): List of mean returns.
        evaluate_freq (int): Evaluation frequency.
        env (str): Environment name (default: 'ALE/Pong-v5').
    """

    evaluation_episodes = [1 + evaluate_freq * index for index in range(len(mean_returns))]
    plt.plot(evaluation_episodes, mean_returns)
    plt.xlabel("Episode")
    plt.ylabel("Mean Return")
    plt.title("Evaluated mean returns")
    if env == 'ALE/Pong-v5':
        os.makedirs('plots/ALE/pong', exist_ok=True)
        plt.savefig("plots/ALE/pong/mean_returns_plot.png")
        # plt.savefig("plots/ALE/pong/mean_returns_plot.eps")
    else:
        os.makedirs('plots/cartpole', exist_ok=True)
        plt.savefig("plots/cartpole/mean_returns_plot.png") 
        # plt.savefig("plots/cartpole/mean_returns_plot.eps")  

def show_plot(env, list_of_configs, all_mean_returns):
    """
    Merge some plots and results in one plot.

    Args:
        env (str): Environment name.
        list_of_configs (list): List of configuration dictionaries.
        all_mean_returns (list): List of mean returns for each configuration.
    """
    colors = ['black', 'blue', 'red', 'green', 'orange', 'purple', 'yellow', 'cyan']
    alpha = 0.6
    evaluation_episodes = [1 + 25 * index for index in range(len(all_mean_returns[0]))]
    # Plot each experiment
    i = 0
    plt.plot(evaluation_episodes, all_mean_returns[i], label=f'Default', linewidth=3, color=colors[i])
    i = 1
    # plt.plot(evaluation_episodes, all_mean_returns[i], label=f'Experiment {i}', alpha=alpha, color=colors[i])
    i = 2
    # plt.plot(evaluation_episodes, all_mean_returns[i], label=f'Experiment {i}', alpha=alpha, color=colors[i])
    i = 3
    plt.plot(evaluation_episodes, all_mean_returns[i], label=f'Experiment {i}', alpha=alpha, color=colors[i])
    i = 4
    plt.plot(evaluation_episodes, all_mean_returns[i], label=f'Experiment {i}', alpha=alpha, color=colors[i])
    i = 5
    # plt.plot(evaluation_episodes, all_mean_returns[i], label=f'Experiment {i}', alpha=alpha, color=colors[i])
    i = 6
    # plt.plot(evaluation_episodes, all_mean_returns[i], label=f'Experiment {i}', alpha=alpha, color=colors[i])
    i = 7
    # plt.plot(evaluation_episodes, all_mean_returns[i], label=f'Experiment {i}', alpha=alpha, color=colors[i])
    
    plt.xlabel("Episode")
    plt.ylabel("Mean Return")
    plt.title("Evaluated mean returns")
    plt.legend()
    plt.show()

