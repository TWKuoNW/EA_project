import gymnasium as gym
import gym_pusht
import time

env = gym.make("gym_pusht/PushT-v0", render_mode="human")

fixed_state = [20.0, 250, 100.0, 200.0, 0.0]  # agent_x,agent_y,block_x,block_y,angle
observation, info = env.reset(options={"reset_to_state": fixed_state})

for _ in range(100000):
    image = env.render()
    input()
    action = env.action_space.sample()
    #action = [0.0, 0.0]
    observation, reward, terminated, truncated, info = env.step(action)
    print("info:", info)
    print("info:", type(info))
    

    if terminated or truncated:
        observation, info = env.reset(options={"reset_to_state": fixed_state})
    
    

env.close()
