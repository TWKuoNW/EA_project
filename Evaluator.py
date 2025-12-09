import math
import time
import gymnasium as gym
import gym_pusht

class Evaluator:
	selfCost = None #cls.selfCost[state[i]]
	selfDamage = None #cls.selfDamage[state[i]]
	EnhanceDamage = None #cls.EnhanceDamage[state[i-1]][state[i]]

	@classmethod
	def ObjFunc(cls, state, render= False):
		# Evaluate the sequence of actions in the PushT environment
		if render == True:
			env = gym.make("gym_pusht/PushT-v0", render_mode="human")
		else:
			env = gym.make("gym_pusht/PushT-v0")

		fixed_state = [20.0, 250, 100.0, 200.0, 0.0]  # agent_x, agent_y, block_x, block_y, angle
		env.reset(options={"reset_to_state": fixed_state})

		rewardEnd = 0.0
		# Each action consists of two consecutive values in the individual
		for i in range(len(state) // 2):
			action = state[2 * i : 2 * i + 2]
			observation, reward, terminated, truncated, info = env.step(action)
			rewardEnd = reward  # keep last reward
			if(render == True):
				env.render()
			if terminated or truncated:
				observation, info = env.reset(options={"reset_to_state": fixed_state})
		env.close()

		nSeqSteps = len(state) // 2

		objectives = [nSeqSteps, rewardEnd]

		return objectives