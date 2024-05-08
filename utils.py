import torch
import csv
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs, env):
    return obs
    # """Performs necessary observation preprocessing."""
    # if env in ['CartPole-v1']:
    #     return torch.tensor(obs, device=device).float()
    # else:
    #     raise ValueError('Please add necessary observation preprocessing instructions to preprocess() in utils.py.')

def transform_action(env, action):
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
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours} hours, {minutes} minutes, {seconds} seconds"

def append_to_csv(config, mean_returns, csv_file = 'mean_return.csv'):
    
    if not os.path.exists(csv_file):
        # If the file doesn't exist, create it and open in write mode
        with open(csv_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(config.keys())+['Mean Return'])
            writer.writeheader()
    mean_returns_str = ','.join(map(str, mean_returns))
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list(config.values()) + [mean_returns_str])

def read_csv(csv_file = 'mean_return.csv'):
    
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
    
def show_plot(env, list_of_configs, all_mean_returns):
    colors = ['black', 'blue', 'red', 'green', 'orange', 'purple']
    alpha = 0.6
    evaluation_episodes = [1 + 25 * index for index in range(len(all_mean_returns[0]))]
    # Plot each experiment
    i = 0
    plt.plot(evaluation_episodes, all_mean_returns[i], label=f'Default', linewidth=3, color=colors[i])
    i = 1
    plt.plot(evaluation_episodes, all_mean_returns[i], label=f'Experiment {i}', alpha=alpha, color=colors[i])
    i = 2
    plt.plot(evaluation_episodes, all_mean_returns[i], label=f'Experiment {i}', alpha=alpha, color=colors[i])
    i = 3
    plt.plot(evaluation_episodes, all_mean_returns[i], label=f'Experiment {i}', alpha=alpha, color=colors[i])
    i = 4
    plt.plot(evaluation_episodes, all_mean_returns[i], label=f'Experiment {i}', alpha=alpha, color=colors[i])
    i = 5
    plt.plot(evaluation_episodes, all_mean_returns[i], label=f'Experiment {i}', alpha=alpha, color=colors[i])
    
   

    plt.xlabel("Episode")
    plt.ylabel("Mean Return")
    plt.title("Evaluated mean returns")
    plt.legend()
    if env == 'ALE/Pong-v5':
        plt.savefig("PONG_mean_returns_plot.png")
        plt.savefig("PONG_mean_returns_plot.eps")
    else:
        plt.savefig("CARTPOLE_mean_returns_plot.png") 
        plt.savefig("CARTPOLE_mean_returns_plot.eps")  
    plt.show()
