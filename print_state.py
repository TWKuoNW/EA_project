import gymnasium as gym
import gym_pusht
import time

env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")

fixed_state = [20.0, 250, 100.0, 200.0, 0.0]  # agent_x,agent_y,block_x,block_y,angle
observation, info = env.reset(options={"reset_to_state": fixed_state})
act = [271.77886431532914, 133.0015390031648, 255.1531122277372, 487.3128775452195, 71.63711920724131, 84.80575641171541, 87.7162411362949, 167.25061642399805, 122.92388939521051, 254.9083133175268, 284.1139707784343, 129.67034867918852, 368.8864628039261, 101.60946817857703, 335.50674995039213, 158.39943668412627, 200.0308653035118, 462.9248550248822, 460.78143087953447, 41.981775263476266]

obs = None
for i in range(act.__len__() // 2):
    image = env.render()
    action = [act[2*i], act[2*i+1]]
    observation, reward, terminated, truncated, info = env.step(action)
    obs = observation
    

    if terminated or truncated:
        observation, info = env.reset(options={"reset_to_state": fixed_state})
print(obs)
#print(env.render())
env.close()
