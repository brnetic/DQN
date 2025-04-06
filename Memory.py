import random
from collections import deque

class Memory():
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

    
    def append(self, state, action, reward, next_state, terminated):
        item = tuple([state,action,reward,next_state,terminated])
        self.memory.append(item)

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        return batch
    
    def size(self):
        return len(self.memory)
    
    

