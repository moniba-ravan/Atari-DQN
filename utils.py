import torch
import csv
import os

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
        reader = csv.reader(file)
        column_names = next(reader)
        for row in reader:
            config = {}
            mean_returns = list()
            for idx, value in enumerate(row):
                if idx == 0:
                    config[column_names[idx]] = value
                elif value.isdigit():
                    config[column_names[idx]] = int(value)
                else:
                    try:
                        config[column_names[idx]] = float(value)
                    except ValueError:
                        mean_returns = list(map(float, value.split(',')))
            print(config)
            print(mean_returns)
            print(mean_returns[0])

