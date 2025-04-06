import torch
from torch import nn
import numpy as np


class DQN(nn.Module):
    def __init__(self,obs_space,n_actions):
        super(DQN,self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(obs_space,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,n_actions)
        )
        

    def forward(self,x):
        #x = x.view(x.size(0), -1)
        

        x= self.l1(x)

        return x
    

class CNN(nn.Module):
    def __init__(self, input_channels,n_actions):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 96, kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 210, 160)
        
            
            cnn_output = self.cnn(dummy_input)
            self.flatten_dim = cnn_output.contiguous().reshape(cnn_output.size(0), -1).shape[1]
            

        self.fc = nn.Sequential(
            
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

   

    def forward(self, x):
        
        x = self.cnn(x)
        x = x.contiguous().reshape(x.size(0), -1)
        return self.fc(x)
