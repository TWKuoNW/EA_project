import gymnasium as gym
import gym_pusht

env = gym.make("gym_pusht/PushT-v0", render_mode="human")

fixed_state = [0.0, 0.0, 100.0, 200.0, 180.0]  # agent_x,agent_y,block_x,block_y,angle
observation, info = env.reset(options={"reset_to_state": fixed_state})

for _ in range(1000):
    action = env.action_space.sample()
    action = [0.0, 0.0]
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()

    if terminated or truncated:
        observation, info = env.reset(options={"reset_to_state": fixed_state})
    

env.close()
