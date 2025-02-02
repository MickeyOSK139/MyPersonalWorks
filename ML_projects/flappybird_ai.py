
import torch
import numpy as np
import pygame


# Ensure that your custom environment class and DQN definition are imported or defined here.
# For example:
from flappybirdtrain import FlappyBirdEnv, DQN

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the environment in render mode to visualize the game.
env = FlappyBirdEnv(render_mode=True)
state_size = 4   # [bird_y, bird_velocity, pipe_dist, pipe_gap_center]
action_size = 2  # 0: do nothing, 1: jump

# Instantiate the model and load the trained weights.
trained_model = DQN(state_size, action_size).to(device)
trained_model.load_state_dict(torch.load('flappy_agent.pth', map_location=device, weights_only=False))
trained_model.eval()  # Set the model to evaluation mode.

# Run the agent in the environment.
state = env.reset()
done = False

while not done:
    # Render the environment to see the agent in action.
    env.render()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    # Prepare the state as a PyTorch tensor.
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    # Obtain the Q-values from the model and choose the action with the highest value.
    with torch.no_grad():
        q_values = trained_model(state_tensor)
    action = q_values.max(1)[1].item()
    
    # Execute the chosen action in the environment.
    state, reward, done, info = env.step(action)



env.close()
