import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.steps_done = 0 # !!
        
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.

        # !! !!
        sample = random.random()
        eps_threshold = self.get_eps_threshold()
        self.steps_done += 1 # ????????????????????????????????????????????????????
        if exploit or sample > eps_threshold: # exploit
            with torch.no_grad(): # Disable gradient computation during exploitation
                return torch.argmax(self.forward(observation), dim=1).unsqueeze(1)
        else: # explore
            return torch.randint(0, self.n_actions, (self.batch_size, 1), device=observation.device) 
            
        # !! !!

    
    def get_eps_threshold(self):
        eps_decay = (self.eps_start - self.eps_end) / self.anneal_length
        return max(self.eps_end, self.eps_start - (self.steps_done * eps_decay) )

    def optimize(dqn, target_dqn, memory, optimizer):
        """This function samples a batch from the replay buffer and optimizes the Q-network."""
        # If we don't have enough transitions stored yet, we don't train.
        if len(memory) < dqn.batch_size:
            return

        # TODO: Sample a batch from the replay memory and concatenate so that there are
        #       four tensors in total: observations, actions, next observations and rewards.
        #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
        #       Note that special care is needed for terminal transitions! 

        observations, actions, next_observations, rewards = memory.sample(dqn.batch_size)
        observations = torch.tensor(observations).to(device)
        actions = torch.tensor(actions).to(device)
        next_observation = torch.tensor(next_observation).to(device)
        rewards = torch.tensor(rewards).to(device)

        if (next_observation == 0).any().item():
            return
        
        # TODO: Compute the current estimates of the Q-values for each state-action
        #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
        #       corresponding to the chosen actions.
        q_values = dqn(observations)
        q_values = torch.gather(q_values, 1, actions.unsqueeze(1))
        # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
        
        # state 0: (-2.4, 2.4), and 2: (-.2095,.2095) otherwise terminate
        next_q_values = torch.max((target_dqn(next_observations)), 1)[0]
        
        # for index, observation in enumerate(observations):
        #     if abs(observation[0]) > 2.4 or abs(observation[2])> 0.2095:
        #         next_q_values[index] = 0
        
        q_value_targets = rewards + target_dqn.gamma*next_q_values
        # Compute loss.
        loss = F.mse_loss(q_values.squeeze(), q_value_targets)

        # Perform gradient descent.
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        return loss.item()

