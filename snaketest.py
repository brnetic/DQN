import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pygame
import matplotlib.pyplot as plt
from collections import deque
import os
import time
import gymnasium as gym
from typing import List, Tuple, Dict, Optional, Union, Any

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Hyperparameters
GRID_SIZE = 6
HIDDEN_SIZE = 512
MEMORY_SIZE = 100000
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.0
EPSILON_DECAY = 0.95
LEARNING_RATE = 0.0001
TARGET_UPDATE_FREQ = 1000
SAVE_FREQ = 500
MAX_EPISODES = 10000
TRAINING_START = 10000  # Start training after this many steps
PRIORITIZED_REPLAY = True
DOUBLE_DQN = True
DUELING_DQN = True

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else 
                     ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

#########################
# Environment Definition
#########################

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, grid_size=GRID_SIZE, render_mode=None, cell_size=30):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.render_mode = render_mode
        
        # Direction constants: 0: RIGHT, 1: DOWN, 2: LEFT, 3: UP
        self.DIRECTIONS = {
            0: (0, 1),   # RIGHT
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (-1, 0)   # UP
        }
        
        # Actions: 0: STRAIGHT, 1: RIGHT, 2: LEFT
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation space: 3 channels (head, body, food) of grid_size x grid_size
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self.grid_size, self.grid_size),
            dtype=np.float32
        )

        self.window = None
        self.clock = None
        self.max_steps_without_food = grid_size * 2  # Allow more steps for larger grids
        self.steps_without_food = 0
        self.total_steps = 0

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Snake starts in the middle
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = random.choice([0, 1, 2, 3])  # Random start direction
        self.spawn_food()
        self.done = False
        self.score = 0
        self.steps_without_food = 0
        self.total_steps = 0
        self.last_distance = self._get_food_distance()
        
        if self.render_mode == "human":
            self.render()
            
        return self.get_observation(), {}

    def spawn_food(self):
        """Spawn food at a random empty position"""
        available_positions = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in self.snake:
                    available_positions.append((r, c))
        
        if available_positions:
            self.food = random.choice(available_positions)
        else:
            # Game is won! (extremely rare)
            self.food = (-1, -1)

    def _get_food_distance(self):
        """Calculate Manhattan distance to food"""
        head_row, head_col = self.snake[0]
        food_row, food_col = self.food
        return abs(head_row - food_row) + abs(head_col - food_col)

    def step(self, action):
        self.total_steps += 1
        self.steps_without_food += 1
        reward = 0
        info = {"score": self.score}
        
        # Store previous game state data for reward shaping
        prev_distance = self._get_food_distance()
        prev_head = self.snake[0]
        
        # Calculate new direction based on action (relative to current direction)
        if action == 0:  # Continue straight
            new_direction = self.direction
        elif action == 1:  # Turn right
            new_direction = (self.direction + 1) % 4
        elif action == 2:  # Turn left
            new_direction = (self.direction - 1) % 4
        
        self.direction = new_direction
        
        # Calculate new head position
        dr, dc = self.DIRECTIONS[self.direction]
        head_row, head_col = self.snake[0]
        new_head = (head_row + dr, head_col + dc)
        new_row, new_col = new_head

        # Check if game is over (collision with wall or self)
        if (
            new_row < 0 or new_row >= self.grid_size or
            new_col < 0 or new_col >= self.grid_size or
            new_head in self.snake
        ):
            self.done = True
            
            # Penalty proportional to distance from food and inversely proportional to snake length
            reward = -10.0 / max(1, len(self.snake) / 2)  
            
            if self.render_mode == "human":
                self.render()
            return self.get_observation(), reward, self.done, False, info

        # Move snake forward
        self.snake.insert(0, new_head)

        # Check if food was eaten
        if new_head == self.food:
            self.score += 1
            info["ate_food"] = True
            
            # Reward for eating food increases with snake length
            reward = 10.0 + len(self.snake) * 0.1  # More reward for longer snake
            
            self.steps_without_food = 0
            self.spawn_food()
        else:
            # Remove tail if no food was eaten
            self.snake.pop()
            
            # Calculate new distance to food for reward shaping
            new_distance = self._get_food_distance()
            
            # Efficient path reward
            if new_distance < prev_distance:
                reward = 0.2  # Reward for moving closer to food
            elif new_distance > prev_distance:
                reward = -0.1  # Small penalty for moving away from food
            else:
                reward = -0.01  # Tiny penalty for not changing distance
            
            # Reward for staying alive but slightly decaying over time without food
            reward += 0.01 * (0.99 ** self.steps_without_food)
            
            # Check for timeout
            if self.steps_without_food >= self.max_steps_without_food * (len(self.snake) + 1):
                reward = -1.0  # Penalty for timeout
                self.done = True
                info["timeout"] = True

        # Additional reward: Positive if snake is not enclosed
        if len(self.snake) > 1:
            spaces_reachable = self.bfs_reachable_empty_tiles()
            max_possible = self.grid_size * self.grid_size - len(self.snake)
            efficiency = spaces_reachable / max_possible if max_possible > 0 else 0
            reward += 0.1 * efficiency  # Small reward for keeping options open

        if self.render_mode == "human":
            self.render()

        return self.get_observation(), reward, self.done, False, info

    def bfs_reachable_empty_tiles(self):
        """Count reachable empty tiles using BFS"""
        visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for r, c in self.snake:
            visited[r][c] = True

        head_row, head_col = self.snake[0]
        queue = [(head_row, head_col)]
        visited[head_row][head_col] = True
        count = 0

        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if not visited[nr][nc]:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
                        count += 1
        return count

    def get_observation(self):
        """Create a 3-channel observation (head, body, food)"""
        head_row, head_col = self.snake[0]
        food_row, food_col = self.food

        # Create separate channels for head, body, and food
        head = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        head[head_row, head_col] = 1

        body = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for i in range(1, len(self.snake)):
            body_row, body_col = self.snake[i]
            body[body_row, body_col] = 1
        
        food = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        if 0 <= food_row < self.grid_size and 0 <= food_col < self.grid_size:
            food[food_row, food_col] = 1

        return np.array([head, body, food], dtype=np.float32)

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Perfect Snake AI")
            self.window = pygame.display.set_mode((self.grid_size * self.cell_size, 
                                                  self.grid_size * self.cell_size))
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            self.window.fill((0, 0, 0))  # Black background

            # Draw grid lines for clarity
            for i in range(self.grid_size):
                pygame.draw.line(self.window, (50, 50, 50), 
                                (0, i * self.cell_size), 
                                (self.grid_size * self.cell_size, i * self.cell_size), 1)
                pygame.draw.line(self.window, (50, 50, 50), 
                                (i * self.cell_size, 0), 
                                (i * self.cell_size, self.grid_size * self.cell_size), 1)

            # Draw snake body
            for i, (row, col) in enumerate(self.snake[1:], 1):
                # Gradient color based on position in body
                color_intensity = max(50, 255 - (i * 5))
                pygame.draw.rect(
                    self.window, 
                    (0, color_intensity, 0),  # Green gradient
                    (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                )
            
            # Draw snake head
            head_row, head_col = self.snake[0]
            pygame.draw.rect(
                self.window, 
                (0, 100, 255),  # Blue head
                (head_col * self.cell_size, head_row * self.cell_size, self.cell_size, self.cell_size)
            )
            
            # Draw food
            food_row, food_col = self.food
            if 0 <= food_row < self.grid_size and 0 <= food_col < self.grid_size:
                pygame.draw.rect(
                    self.window, 
                    (255, 0, 0),  # Red food
                    (food_col * self.cell_size, food_row * self.cell_size, self.cell_size, self.cell_size)
                )
            
            # Display score and length
            font = pygame.font.SysFont('arial', 16)
            score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
            length_text = font.render(f'Length: {len(self.snake)}', True, (255, 255, 255))
            self.window.blit(score_text, (5, 5))
            self.window.blit(length_text, (5, 25))

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            
        elif self.render_mode == "rgb_array":
            # Create an RGB array for machine consumption
            canvas = np.zeros((self.grid_size * self.cell_size, 
                              self.grid_size * self.cell_size, 3), dtype=np.uint8)
            
            # Draw snake body
            for row, col in self.snake[1:]:
                canvas[row*self.cell_size:(row+1)*self.cell_size, 
                       col*self.cell_size:(col+1)*self.cell_size] = [0, 255, 0]
                
            # Draw snake head
            head_row, head_col = self.snake[0]
            canvas[head_row*self.cell_size:(head_row+1)*self.cell_size, 
                   head_col*self.cell_size:(head_col+1)*self.cell_size] = [0, 100, 255]
                
            # Draw food
            food_row, food_col = self.food
            if 0 <= food_row < self.grid_size and 0 <= food_col < self.grid_size:
                canvas[food_row*self.cell_size:(food_row+1)*self.cell_size, 
                       food_col*self.cell_size:(food_col+1)*self.cell_size] = [255, 0, 0]
                
            return canvas

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

#########################
# Memory Buffer
#########################

class PrioritizedReplayBuffer:
    def __init__(self, capacity, batch_size, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha  # How much prioritization to use (0 = no prioritization)
        self.beta = beta    # Importance sampling weight
        self.beta_increment = beta_increment  # Beta annealing
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.epsilon = 1e-5  # Small number to avoid zero priorities
    
    def add(self, state, action, reward, next_state, done):
        # Set max priority for new experience
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self):
        if len(self.buffer) < self.batch_size:
            return None
            
        # Increase beta over time for more accurate corrections
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= np.sum(probabilities)
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probabilities)
        
        # Calculate importance-sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)  # Normalize
        
        # Get batch and weights
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
            
        batch = (
            torch.tensor(np.array(states), dtype=torch.float32).to(device),
            torch.tensor(np.array(actions), dtype=torch.long).to(device),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(device),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(device)
        )
        
        weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)
        return batch, indices, weights_tensor
    
    def update_priorities(self, indices, td_errors):
        for i, idx in enumerate(indices):
            self.priorities[idx] = abs(td_errors[i]) + self.epsilon
    
    def size(self):
        return len(self.buffer)

class StandardReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self):
        if len(self.buffer) < self.batch_size:
            return None
            
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(device),
            torch.tensor(np.array(actions), dtype=torch.long).to(device),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(device),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(device)
        ), None, None
    
    def update_priorities(self, indices, td_errors):
        pass  # Not used in standard buffer
    
    def size(self):
        return len(self.buffer)

#########################
# DQN Models
#########################

class CNNModel(nn.Module):
    """Standard DQN with convolutional layers"""
    def __init__(self, input_channels, num_actions, grid_size):
        super(CNNModel, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Calculate output size after convolutions
        conv_output_size = grid_size // 2  # After stride 2
        self.flatten_dim = 64 * conv_output_size * conv_output_size
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, num_actions)
        )
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DuelingCNNModel(nn.Module):
    """Dueling DQN with convolutional layers"""
    def __init__(self, input_channels, num_actions, grid_size):
        super(DuelingCNNModel, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Calculate output size after convolutions
        conv_output_size = grid_size // 2  # After stride 2
        self.flatten_dim = 64 * conv_output_size * conv_output_size
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.flatten_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, num_actions)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(self.flatten_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )
    
    def forward(self, x):
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        
        advantage = self.advantage_stream(features)
        value = self.value_stream(features)
        
        # Combine value and advantage using the dueling trick
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

#########################
# DQN Agent
#########################

class DQNAgent:
    def __init__(self, input_channels, num_actions, grid_size):
        self.num_actions = num_actions
        self.step_counter = 0
        self.epsilon = EPSILON_START
        
        # Set up memory buffer
        self.memory = (PrioritizedReplayBuffer(MEMORY_SIZE, BATCH_SIZE) 
                      if PRIORITIZED_REPLAY else 
                      StandardReplayBuffer(MEMORY_SIZE, BATCH_SIZE))
        
        # Set up DQN models
        if DUELING_DQN:
            self.policy_net = DuelingCNNModel(input_channels, num_actions, grid_size).to(device)
            self.target_net = DuelingCNNModel(input_channels, num_actions, grid_size).to(device)
        else:
            self.policy_net = CNNModel(input_channels, num_actions, grid_size).to(device)
            self.target_net = CNNModel(input_channels, num_actions, grid_size).to(device)
            
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network stays in eval mode
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')  # Huber loss with no reduction for PER
        
        # For logging
        self.losses = []
        self.avg_q_values = []
        
    def select_action(self, state, evaluate=False):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        
        with torch.no_grad():
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
            q_values = self.policy_net(state_tensor)
            self.avg_q_values.append(q_values.mean().item())
            return q_values.argmax(1).item()
        
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
    
    def update_model(self):
        self.step_counter += 1
        
        # Don't update until we have enough data
        if self.memory.size() < BATCH_SIZE or self.memory.size() < TRAINING_START:
            return 0
        
        # Sample from memory
        batch_data = self.memory.sample()
        if batch_data is None:
            return 0
            
        if PRIORITIZED_REPLAY:
            batch, indices, weights = batch_data
        else:
            batch, indices, weights = batch_data
            weights = torch.ones((BATCH_SIZE, 1)).to(device)  # Uniform weights
            
        states, actions, rewards, next_states, dones = batch
        
        # Compute current Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values based on algorithm (Double DQN or standard)
        with torch.no_grad():
            if DOUBLE_DQN:
                # Double DQN: Select actions using policy net, evaluate using target net
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                # Standard DQN: Select and evaluate using target net
                next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            
            # Compute target Q values
            target_q_values = rewards + (1 - dones) * GAMMA * next_q_values
        
        # Compute loss with importance sampling weights from prioritized replay
        td_errors = target_q_values - q_values
        loss = self.loss_fn(q_values, target_q_values) * weights
        loss = loss.mean()
        
        # Update priorities in memory
        if PRIORITIZED_REPLAY:
            self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        
        self.optimizer.step()
        
        # Periodically update target network
        if self.step_counter % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # Log training info
        self.losses.append(loss.item())
        
        return loss.item()
    
    def save_model(self, path):
        """Save the model and training state"""
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_counter': self.step_counter,
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load the model and training state"""
        if not os.path.exists(path):
            print(f"No model found at {path}")
            return False
            
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_counter = checkpoint['step_counter']
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {path}")
        return True

#########################
# Training Functions
#########################

def plot_training_progress(episode_rewards, avg_rewards, losses, avg_q_values, epsilon_values, filename="training_metrics.png"):
    """Plot training metrics"""
    plt.figure(figsize=(18, 12))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards, label='Episode Reward', alpha=0.6)
    plt.plot(avg_rewards, label='100-episode Average', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    
    # Plot losses
    if losses:
        plt.subplot(2, 2, 2)
        plt.plot(losses, alpha=0.6)
        # Plot smoothed losses
        if len(losses) > 100:
            smoothed = np.convolve(losses, np.ones(100)/100, mode='valid')
            plt.plot(smoothed, color='red')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.yscale('log')
    
    # Plot average Q values
    if avg_q_values:
        plt.subplot(2, 2, 3)
        plt.plot(avg_q_values, alpha=0.6)
        # Plot smoothed Q values
        if len(avg_q_values) > 100:
            smoothed = np.convolve(avg_q_values, np.ones(100)/100, mode='valid')
            plt.plot(smoothed, color='red')
        plt.xlabel('Action Selection')
        plt.ylabel('Average Q Value')
        plt.title('Q Value Estimates')
    
    # Plot epsilon values
    if epsilon_values:
        plt.subplot(2, 2, 4)
        plt.plot(epsilon_values)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def train_agent(env, agent, episodes, model_dir="models"):
    """Train the DQN agent"""
    # Create directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Tracking metrics
    all_rewards = []
    avg_rewards = []
    epsilon_values = []
    best_avg_reward = -float('inf')
    reward_window = deque(maxlen=100)
    
    for episode in range(1, episodes + 1):
        start_time = time.time()
        state, _ = env.reset(seed=episode)  # Different seed each episode
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            
            # Store transition in memory
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Train the network
            loss = agent.update_model()
        
        # Record metrics
        reward_window.append(episode_reward)
        all_rewards.append(episode_reward)
        avg_reward = sum(reward_window) / len(reward_window)
        avg_rewards.append(avg_reward)
        epsilon_values.append(agent.epsilon)
        
        # Print progress
        duration = time.time() - start_time
        print(f"Episode {episode}/{episodes} | Reward: {episode_reward:.2f} | " +
              f"Avg Reward: {avg_reward:.2f} | Steps: {steps} | " +
              f"Snake Length: {info['score'] + 1} | Epsilon: {agent.epsilon:.4f} | " +
              f"Duration: {duration:.2f}s")
        
        # Save model periodically and when we achieve a new best average
        if episode % SAVE_FREQ == 0:
            agent.save_model(f"{model_dir}/snake_dqn_episode_{episode}.pt")
            plot_training_progress(all_rewards, avg_rewards, agent.losses, 
                                  agent.avg_q_values, epsilon_values)
        
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save_model(f"{model_dir}/snake_dqn_best.pt")
    
    # Final save
    agent.save_model(f"{model_dir}/snake_dqn_final.pt")
    plot_training_progress(all_rewards, avg_rewards, agent.losses, 
                          agent.avg_q_values, epsilon_values)
    
    return all_rewards, avg_rewards, agent.losses, agent.avg_q_values

def evaluate_agent(env, agent, episodes=10, render=True):
    """Evaluate the trained agent"""
    rewards = []
    lengths = []
    
    for episode in range(episodes):
        state, _ = env.reset(seed=episode + 1000)  # Different seeds from training
        done = False
        episode_reward = 0
        
        while not done:
            # No exploration during evaluation
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _, info = env.step(action)
            state = next_state
            episode_reward += reward
        
        rewards.append(episode_reward)
        lengths.append(info['score'] + 1)  # Snake length is score + 1
        print(f"Evaluation Episode {episode+1}/{episodes} | " +
              f"Reward: {episode_reward:.2f} | Snake Length: {info['score'] + 1}")
    
    avg_reward = sum(rewards) / len(rewards)
    avg_length = sum(lengths) / len(lengths)
    print(f"Average Evaluation Reward: {avg_reward:.2f}")
    print(f"Average Snake Length: {avg_length:.2f}")
    
    return rewards, lengths

#########################
# Main Function
#########################

def main():
    # Create environment
    render_mode = "human" if GRID_SIZE <= 20 else None  # Only render for smaller grids
    env = SnakeEnv(grid_size=GRID_SIZE, render_mode=None)
    
    # Create agent
    agent = DQNAgent(
        input_channels=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        grid_size=GRID_SIZE
    )
    
    # Check for existing model to continue training
    model_path = "models/snake_dqn_best.pt"
    if os.path.exists(model_path):
        print("Found existing model. Do you want to continue training? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            agent.load_model(model_path)
    
    # Training phase
    print("\n=== Starting Training ===")
    print(f"Device: {device}")
    print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Using Prioritized Replay: {PRIORITIZED_REPLAY}")
    print(f"Using Double DQN: {DOUBLE_DQN}")
    print(f"Using Dueling DQN: {DUELING_DQN}\n")
    
    train_agent(env, agent, MAX_EPISODES)
    
    # Evaluation phase with rendering
    print("\n=== Starting Evaluation ===")
    eval_env = SnakeEnv(grid_size=GRID_SIZE, render_mode="human")
    evaluate_agent(eval_env, agent, episodes=5)
    
    # Close environments
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()