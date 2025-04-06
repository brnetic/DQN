from DQN import DQN
from DQN import CNN
from Memory import Memory
from torch import optim
import torch
import numpy as np


class Agent():
    def __init__(self,lr,gamma,mem_size,batch_size,state_dim,action_dim,eps,eps_decay,eps_min):
        self.memory = Memory(memory_size=mem_size,batch_size=batch_size)
        self.lr = lr
        self.gamma = gamma
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.batch_size = batch_size
        
        self.device = torch.device('mps')

        self.policy_network = CNN(3,action_dim).to(self.device)
        self.target_network = CNN(3,action_dim).to(self.device)
        #self.policy_network = DQN(obs_space=state_dim,n_actions=action_dim).to(self.device)
        #self.target_network = DQN(obs_space=state_dim,n_actions=action_dim).to(self.device)

        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.step = 0

        self.lossfn = torch.nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.policy_network.parameters(),lr = self.lr)

    def select_action(self,state):
        self.step+=1
        self.eps *= self.eps_decay
        self.eps = max(self.eps,self.eps_min)
        if np.random.random()<self.eps:
            
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.tensor(state,dtype=torch.float32,device=self.device)
            state_tensor = state_tensor.unsqueeze(0)
            
        
            with torch.no_grad():
                out = self.policy_network(state_tensor)
            return torch.argmax(out).item()


        
    def train(self):
        if self.memory.size() < 1000:
            return

        batch = self.memory.sample()
        states, actions, rewards, next_states, dones = zip(*batch)

        

        states = torch.tensor(np.array(states),dtype=torch.float32,device=self.device)
        
        next_states = torch.tensor(np.array(next_states),dtype=torch.float32,device=self.device)
        
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)


        # Q-learning update
        q_values = self.policy_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.lossfn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        


        if self.step >= 1000:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.step = 0

    def store_transition(self,state,action,next_state,reward,terminated):

        self.memory.append(state=state,action=action,reward=reward,next_state=next_state,terminated=terminated)

    def save_model(self):
        torch.save(self.policy_network, 'DQN_model.pth')

    def load_model(self, path='DQN_model.pth'):
        self.policy_network = torch.load(path)
        self.policy_network.eval()
        self.target_network.load_state_dict(self.policy_network.state_dict())

