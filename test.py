from env import SnakeEnv
import torch



cnn = torch.load("DQN_model.pth", weights_only=False)
cnn.eval()

env = SnakeEnv(grid_size=10, render_mode="human")

done = False
state, _ = env.reset()
device = torch.device('mps')

while not done:
    with torch.no_grad():

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state_tensor = state_tensor.to(device)

        action = cnn(state_tensor)
        action = torch.argmax(action).item()

    next_state, reward, terminated, truncated, _ = env.step(action)
    
    done = terminated or truncated
    state = next_state


    