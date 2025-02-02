import random
import math
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pygame
import matplotlib.pyplot as plt
import pandas as pd
from FlappyBird import Bird

# ================================
# 1. Custom Flappy Bird Environment
# ================================

# The environment mimics the Flappy Bird clone described earlier.
# We define state, reward, and termination conditions analogous to the game.
# The state is represented as a vector: [bird_y, bird_velocity, pipe_dist, pipe_gap_center].
# The action space is discrete: 0 (do nothing) and 1 (jump).

class FlappyBirdEnv:
    def __init__(self, render_mode=False):
        # Screen dimensions and initial game settings.
        self.WIDTH, self.HEIGHT = 400, 600
        self.win = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Flappy Bird Dynamic Difficulty")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("comicsans", 40)
        try:
            raw_bird_image = pygame.image.load("birb.png").convert_alpha()
        except pygame.error as e:
            print("Error loading bird.png:", e)
            raise SystemExit
        self.bird_sprite = pygame.transform.scale(raw_bird_image, (34, 24))
        
        # Bird constants (position, physics, etc.)
        self.BIRD_X = 50
        self.bird_y = self.HEIGHT / 2
        self.bird_velocity = 0
        
        # Base pipe parameters.
        self.BASE_PIPE_VELOCITY = -3      # Starting pipe velocity.
        self.BASE_PIPE_GAP = 200          # Starting gap size.
        self.MIN_PIPE_GAP = 80            # Minimum gap allowed.
        self.base_spawn_interval = 90     # Base spawn interval in frames when pipe velocity is BASE_PIPE_VELOCITY.

        
        # Difficulty adjustments (applied every 10 points).
        self.velocity_increment = -0.2    # Increase speed (more negative makes pipes move faster).
        self.gap_decrement = 5           # Decrease gap size.
        
        # Initialize current difficulty values.
        self.current_pipe_velocity = self.BASE_PIPE_VELOCITY
        self.current_pipe_gap = self.BASE_PIPE_GAP
        
        # Pipe settings.
        self.PIPE_WIDTH = 52
        self.pipes = []
        self.pipe_spawn_interval = 90  # Frames between spawning new pipes.
        self.frame_count = 0
        self.time_since_last_pipe = 0
        self.base_spawn_interval = 90 
        
        # Score.
        self.score = 0
        
        # Render mode flag.
        self.render_mode = render_mode
        
        self.reset()

    def reset(self):
        self.bird_y = self.HEIGHT / 2
        self.bird_velocity = 0
        self.pipes = []
        self.frame_count = 0
        self.score = 0
        # Spawn the first pipe.
        self._spawn_pipe()
        return self._get_state()

    def _spawn_pipe(self):
        # Randomly choose a gap center. The gap size will be determined dynamically.
        gap_y = random.randint(100, self.HEIGHT - 100)
        pipe = {'x': self.WIDTH, 'gap_y': gap_y, 'scored': False}
        self.pipes.append(pipe)

    def _update_difficulty(self):
        # Update difficulty parameters every step based on current score.
        difficulty_level = self.score // 10  # Increase difficulty every 10 points.
        self.current_pipe_velocity = self.BASE_PIPE_VELOCITY + difficulty_level * self.velocity_increment
        self.current_pipe_gap = max(self.BASE_PIPE_GAP - difficulty_level * self.gap_decrement, self.MIN_PIPE_GAP)

    def _get_state(self):
        # For illustration, the state consists of normalized bird position, velocity, and the next pipe parameters.
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + self.PIPE_WIDTH >= self.BIRD_X:
                next_pipe = pipe
                break
        if next_pipe is None:
            next_pipe = {'x': self.WIDTH, 'gap_y': self.HEIGHT / 2, 'scored': False}
        pipe_dist = next_pipe['x'] + self.PIPE_WIDTH - self.BIRD_X
        pipe_gap_center = next_pipe['gap_y']
        state = np.array([self.bird_y / self.HEIGHT,
                          self.bird_velocity / 10.0,
                          pipe_dist / self.WIDTH,
                          pipe_gap_center / self.HEIGHT],
                         dtype=np.float32)
        return state

    def step(self, action):
        # Update difficulty based on the current score.
        self._update_difficulty()

        # Process the action: if jump, set bird velocity.
        if action == 1:
            self.bird_velocity = -10  # JUMP_VELOCITY

        # Apply gravity and update the bird's vertical position.
        self.bird_velocity += 0.5  # GRAVITY
        self.bird_y += self.bird_velocity

        # Update pipe positions using the current difficulty (pipe velocity).
        for pipe in self.pipes:
            pipe['x'] += self.current_pipe_velocity

        # Increment the time accumulator.
        self.time_since_last_pipe += 1

        # Calculate dynamic spawn interval based on current pipe velocity.
        # Using absolute values ensures we work with positive intervals.
        dynamic_spawn_interval = int(self.base_spawn_interval * (abs(self.BASE_PIPE_VELOCITY) / abs(self.current_pipe_velocity)))
        # Clamp to a minimum value to avoid excessively rapid spawning.
        dynamic_spawn_interval = max(dynamic_spawn_interval, 60)

        # Check if enough time has elapsed to spawn a new pipe.
        if self.time_since_last_pipe >= dynamic_spawn_interval:
            self._spawn_pipe()
            # Subtract the interval rather than resetting to 0 to preserve any extra accumulated frames.
            self.time_since_last_pipe -= dynamic_spawn_interval

        # Reward structure and score updating.
        reward = 0.1  # Small reward for staying alive.
        for pipe in self.pipes:
            if not pipe['scored'] and (pipe['x'] + self.PIPE_WIDTH < self.BIRD_X):
                pipe['scored'] = True
                self.score += 1
                reward += 10

        # Check for collisions.
        done = False
        if self.bird_y <= 0 or self.bird_y >= self.HEIGHT:
            done = True
            reward = -20

        bird_rect = pygame.Rect(self.BIRD_X - 17, self.bird_y - 12, 34, 24)
        for pipe in self.pipes:
            top_pipe_rect = pygame.Rect(pipe['x'], 0, self.PIPE_WIDTH, pipe['gap_y'] - self.current_pipe_gap / 2)
            bottom_pipe_rect = pygame.Rect(pipe['x'], pipe['gap_y'] + self.current_pipe_gap / 2,
                                        self.PIPE_WIDTH, self.HEIGHT - (pipe['gap_y'] + self.current_pipe_gap / 2))
            if bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect):
                done = True
                reward = -10
                break

        # Remove pipes that have moved off-screen.
        self.pipes = [pipe for pipe in self.pipes if pipe['x'] + self.PIPE_WIDTH > 0]
        next_state = self._get_state()
        info = {'score': self.score}

        if self.render_mode:
            self.render()

        return next_state, reward, done, info

    def render(self):
        # Rendering function using Pygame.
        self.win.fill((135, 206, 250))  # sky-blue background

        # Draw pipes using the current difficulty parameters.
        for pipe in self.pipes:
            # Top pipe.
            top_pipe_height = pipe['gap_y'] - self.current_pipe_gap / 2
            pygame.draw.rect(self.win, (0, 255, 0), (pipe['x'], 0, self.PIPE_WIDTH, top_pipe_height))
            # Bottom pipe.
            bottom_pipe_y = pipe['gap_y'] + self.current_pipe_gap / 2
            pygame.draw.rect(self.win, (0, 255, 0), (pipe['x'], bottom_pipe_y, self.PIPE_WIDTH, self.HEIGHT - bottom_pipe_y))
        
        # --- Draw bird with sprite and rotation based on its y velocity ---
        # Compute the rotation angle. Here, a scaling factor of 3 is used.
        angle = -self.bird_velocity * 3
        # Clamp the angle to between +30 (upward tilt) and -60 (downward tilt).
        angle = max(min(angle, 30), -60)
        
        # Rotate the original bird sprite by the computed angle.
        # It is assumed that self.bird_sprite holds the original, scaled bird image.
        rotated_bird = pygame.transform.rotate(self.bird_sprite, angle)
        
        # Re-center the rotated image at the bird's current position.
        bird_rect = rotated_bird.get_rect(center=(self.BIRD_X, self.bird_y))
        self.win.blit(rotated_bird, bird_rect)
        
        # Draw score.
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.win.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        pygame.display.update()
        self.clock.tick(60)

    def close(self):
        pygame.quit()

# ================================
# 2. DQN Agent with PyTorch
# ================================

# We implement a DQN agent following Mnih et al. (2013). The neural network takes as input
# the state vector (of dimension 4) and outputs Q-values for the two possible actions.
# An experience replay buffer is used to decorrelate transitions.

# Define a simple Multi-Layer Perceptron (MLP) for the Q-network.
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define a transition tuple and the replay memory.
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define the DQN agent.
class DQNAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayMemory(10000)

        self.batch_size = 64
        self.gamma = 0.99  # discount factor
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 10000  # decay steps
        self.steps_done = 0

    def select_action(self, state):
        # Epsilon-greedy action selection.
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if random.random() < eps_threshold:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors.
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # Compute Q(s, a) using the policy network.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s') for all next states using the target network.
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        # If done, then no next state value.
        expected_state_action_values = reward_batch + (1 - done_batch) * self.gamma * next_state_values

        # Compute loss (MSE).
        loss = F.mse_loss(state_action_values.squeeze(), expected_state_action_values)

        # Optimize the model.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ================================
# 3. Training Loop
# ================================

def train_dqn(num_episodes=500, render=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FlappyBirdEnv(render_mode=render)
    state_size = 4   # [bird_y, bird_velocity, pipe_dist, pipe_gap_center]
    action_size = 2  # 0: do nothing, 1: jump
    agent = DQNAgent(state_size, action_size, device)
    target_update = 10  # Update target network every 10 episodes

    episode_rewards = []
    for i_episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            # Store the transition in memory.
            agent.memory.push(state, action, reward, next_state, float(done))
            state = next_state
            # Perform one step of optimization.
            agent.optimize_model()
        episode_rewards.append(total_reward)
        # Update the target network periodically.
        if i_episode % target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print(f"Episode {i_episode}: Total Reward: {total_reward}, Score: {info['score']}")

    env.close()
    return agent, episode_rewards

if __name__ == '__main__':
    # Train the agent. To see rendering, set render=True (note that rendering slows training).
    num_episodes = 10000
    trained_agent, rewards = train_dqn(num_episodes=num_episodes, render=False)
    # After training, export the trained model parameters to disk.
    df = pd.DataFrame(rewards)
    df.to_csv('rewards.csv', index=False)
    torch.save(trained_agent.policy_net.state_dict(), 'flappy_agent.pth')
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(np.arange(1, num_episodes+1),rewards, s=2, c='k')
    fig.savefig("rewards.png")