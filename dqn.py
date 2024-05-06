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

    def push(self, obs, action, next_obs, reward, terminated): # adding terminated argument
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward, terminated)
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
        # self.obs_stack_size = env_config["obs_stack_size"]
        self.env = env_config["env_name"]
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]
        self.decay_rate = (self.eps_start-self.eps_end)/self.anneal_length
        self.eps_threshold = self.eps_start
        self.update_count = 0

        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers 
        # For Cartpole
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)
        # For PONG
        self.fc3 = nn.Linear(3136, 512)  # Adjust the input features according to your input size calculation
        self.fc4 = nn.Linear(512, self.n_actions)

    def forward(self, x):
        if self.env == 'ALE/Pong-v5':
            # Apply convolutional layers with ReLU activations
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))

            # Flatten the output from the conv layers
            x = self.flatten(x)
            # print(f"size: {x.size()}")
            # Apply fully connected layers
            x = F.relu(self.fc3(x))
            x = self.fc4(x)  # No activation function here as this is the output layer
        elif self.env == 'CartPole-v1':
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        return x
    
    def update_eps_threshold(self):
        self.eps_threshold = max(self.eps_end, self.eps_threshold - self.decay_rate)
        self.update_count += 1
        # print(f'Updated epsilon to: {self.eps_threshold}')
        return

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.
        sample = random.random()  # Generate a random sample for epsilon comparison
        # print(f"sample: {sample}")
        if exploit or sample > self.eps_threshold:
            with torch.no_grad():  # No gradient calculation needed here
                q_values = self.forward(observation)
                action = torch.argmax(q_values).unsqueeze(0) # Best action (max Q-value)
        else:
            action = torch.tensor(random.choice([0,1]), device=observation.device)  # Random action

        return action

def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    observations, actions, next_observations, rewards, terminated = memory.sample(dqn.batch_size)
    # tuple(32 of tensors [ 1, 4, 84, 84])
    observations = torch.stack(list(observations)).to(device)
    actions = torch.tensor(actions, device=device).unsqueeze(-1)
    next_observations = torch.stack(list(next_observations))
    rewards = torch.tensor(rewards, device=device)
    terminated = torch.tensor(terminated, dtype=torch.bool, device=device)
    # print(f"obs: {observations.size()}")
    # print(f"acitons: {actions.size()}")
    # print(f"next_obs: {next_observations.size()}")
    # print(f"rewards: {rewards.size()}")
    # print(f"terminated: {terminated.size()}")
    
    # Current Q values are estimated by the main network
    # print(f"dqn: {dqn(observations).size()}")
    current_q_values = dqn(observations).gather(1, actions)
    # print(f"cuurent: {current_q_values}")
    # Next Q values are estimated by the target network
    next_q_values = target_dqn(next_observations).max(1)[0].detach()
    # print(f"next_q {terminated}")
    next_q_values[terminated] = 0  # Zero Q value for terminated states

    # Compute the target of the current Q values
    target_q_values = rewards + (target_dqn.gamma * next_q_values)

    # Compute the loss (Mean Squared Error)
    loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

    # Perform gradient descent
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(dqn.parameters(), 1) # Clip 
    optimizer.step()

    dqn.update_eps_threshold() # !!!!!!!!!!!!!!!!!!!

    return loss.item()        
 

