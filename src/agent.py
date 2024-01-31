import os, random, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from collections import deque
from constants import (
    RESIZE_WINDOW_WIDTH, 
    RESIZE_WINDOW_HEIGHT, 
    NETWORK_MODEL_PATH, 
    NETWORK_MODEL_NAME,
    NETWORK_LEARNING_RATE,
    REPLAY_BUFFER_SIZE,
    BATCH_SIZE,
    DIRECTION_LIST,
    REWARD_DISCOUNT_GAMMA
    )

class Agent(nn.Module):
    def __init__(self, input_channels=3, output_size=4) -> None:
        """
        Initializes the agent

        Parameters
        ----------
        input_channels : int, default = 3
            number of channels, by default assuming RGB color channels
        output_size : int, default = 4
            output probabilities, by default 4, assuming action space consisting of 4 actions - left, up, right, down
        """
        
        # Inherit the functionality of nn.Module
        super(Agent, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=int(64*(RESIZE_WINDOW_WIDTH/4)*(RESIZE_WINDOW_HEIGHT/4)), out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=output_size)

        # Input image : (RESIZE_WINDOW_WIDTH, RESIZE_WINDOW_HEIGHT, input_channels)
        # After conv1 : (RESIZE_WINDOW_WIDTH, RESIZE_WINDOW_HEIGHT, 32)
        # After max_pool_2d : (RESIZE_WINDOW_WIDTH / 2, RESIZE_WINDOW_HEIGHT / 2, 32)
        # After conv2 : (RESIZE_WINDOW_WIDTH / 2, RESIZE_WINDOW_HEIGHT / 2, 64)
        # After max_pool_2d : (RESIZE_WINDOW_WIDTH / 4, RESIZE_WINDOW_HEIGHT / 4, 64)

        self.optimizer = optim.Adam(self.parameters(), lr=NETWORK_LEARNING_RATE)
        self.pre_process_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((RESIZE_WINDOW_WIDTH, RESIZE_WINDOW_HEIGHT)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ]) 

        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    def store_experience(self, experience) -> None:
        self.replay_buffer.append(experience)

    def sample_experience_batch(self) -> list[tuple]:
        # There's a lot of debate in the internet stating that randomly sampling a minibatch reduces variance and correlation in updates
        # From the other point of view, temporal sequence of events can be beneficial to identify dynamics of the environment
        # thus the most appropriate way if sampling a batch should be taken a look again at

        # return random.sample(self.replay_buffer, min(len(self.replay_buffer), BATCH_SIZE))
        return list(self.replay_buffer)[-min(len(self.replay_buffer), BATCH_SIZE):]

    def save_model(self) -> None:
        """
        Saves learned agent weights and parameters, creates model directory if not exists
        """

        if (not os.path.exists(NETWORK_MODEL_PATH)):
            os.makedirs(NETWORK_MODEL_PATH)

        torch.save(self.state_dict(), os.path.join(NETWORK_MODEL_PATH, NETWORK_MODEL_NAME))

    def load_model(self) -> None:
        """
        Loads saved parameters if model file found
        """

        model_file_path = os.path.join(NETWORK_MODEL_PATH, NETWORK_MODEL_NAME)

        if (os.path.exists(model_file_path)):
            self.load_state_dict(torch.load(model_file_path))

    def forward(self, x) -> tuple:
        """
        Feeds the input to the underlying CNN network 

        Returns
        -------
        probabilities : tuple 
            tuple of determined probabilities for all of the actions in the action space
        """

        # Feature detection part
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten()

        # Clasifier part
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)

        return x

    def best_action(self, probabilities) -> tuple[int, int]:
        """
        Returns appropriately formated best action based on given probability distribution

        Returns
        -------
        tuple[int, int]
            index of the highest probable action, formated corresponding game state action
        """

        argmax = torch.argmax(probabilities).item()

        return argmax, DIRECTION_LIST[argmax].value
    
    def sample_action(self, probabilities) -> tuple[int, int]:
        """
        Samples random action to perform

        Returns
        -------
        tuple[int, int]
            index of the sampled action, formated corresponding game state action
        """

        distribution = torch.distributions.Categorical(probabilities)
        sample = distribution.sample().item()

        return sample, DIRECTION_LIST[sample].value

    def calculate_discounted_rewards(self, rewards) -> np.ndarray:
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0

        # Calculate discounted rewards in reverse order
        for timestep in reversed(range(0, discounted_rewards.size)):
            running_add = running_add * REWARD_DISCOUNT_GAMMA + rewards[timestep]
            discounted_rewards[timestep] = running_add 

        # Normalize the discounted rewards and add a very small value for edge case when std is 0
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        return discounted_rewards

    def pre_process(self, pixel_array) -> torch.FloatTensor:
        """
        Performs resizing and normalization of the game's pixel array 

        Parameters
        ----------
        pixel_array : numpy.ndarray
            Numpy array containing games pixels
        """

        # Transposes the array to match PyTorch's channel-first format, downsamples the pixel array, grayscales it and normalizes to be in range [0, 1].
        # In PyTorch, the expected order for a 3D image tensor is (batch_size, channels, height, width), thus
        # we transpose the matrix in a way that the original third dimension (RGB channels, notated by 2) become the first element and so on...
        pixel_array = self.pre_process_transform(torch.from_numpy(np.transpose(pixel_array, (2, 0, 1))))

        # Create a tensor from the pixel array and add the 0th dimension,
        # because as mentioned above, tensors first dimension is batch_size, this way it is initialized to (batch_size=1, channels, height, width)
        return pixel_array.unsqueeze(0)
    
    def train(self) -> None:
        # Collect batch of training samples
        sampled_training_batch = self.sample_experience_batch()
        sampled_observations, sampled_action_idxs, sampled_rewards = zip(*sampled_training_batch)

        # Compute discounted reward for each sample
        discounted_rewards = self.calculate_discounted_rewards(sampled_rewards)

        # Initialize policy loss
        policy_loss = torch.zeros(1)
        
        # Go over the experiences
        for s_obs, s_a_id, d_reward in zip(sampled_observations, sampled_action_idxs, discounted_rewards):
            # For each sample calculate the probability of the action that was actually taken for that state
            s_action_probabilities = self(s_obs)
            performed_s_action_probability = s_action_probabilities[s_a_id]

            # Multiply the log probability with the discounted return, we do this because it simplifies math and prevents floating point errors
            log_prob = torch.log(performed_s_action_probability)

            # We want to maximize this, but since neural nets work by minimizing a loss, we take its negative, which achieves the same effect. Finally, add up these negative quantities for all states.
            policy_loss += -log_prob * d_reward

        # Update network parameters
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()