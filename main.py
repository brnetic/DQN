from Agent import Agent
import gymnasium as gym
import ale_py
import numpy as np
import torch



    

if __name__ == "__main__":
    env = gym.make("CarRacing-v3",render_mode="human")

    print(env.action_space)

    agent = Agent(mem_size=20000,lr=1e-4,gamma=0.99,batch_size=32,eps=0.95,eps_decay=0.995,eps_min=0.05,action_dim=5,state_dim=8)

    best_reward = 0

    for i in range(100000):
        done = False
        state = env.reset()

        state = np.array(state)
        #state = state.transpose(2,0,1)
        total_reward = 0
 
        while not done:
            
            
            
            action = agent.select_action(state)
            #print(f"Action: {action}")
            
            next_state, reward, terminated, _ = env.step(action)
            done = terminated 

            next_state = np.array(next_state)
            #next_state = next_state.transpose(2,0,1)

            agent.store_transition(state, action, next_state, reward, done)
            agent.train()
            state = next_state

            total_reward += reward
        
        if total_reward>best_reward:
            best_reward = total_reward
            agent.save_model()
            print(f"Reached best reward so far saving model, reward: {best_reward}")

        print(f"episode: {i}, reward:{total_reward}")
    agent.save_model()
    
    
