import gymnasium as gym
import gym_pusht
import random
import time

# -------------------------------------------------------------
# Fitness function
# -------------------------------------------------------------
def fitness_function(individual, render="rgb_array"): # human rgb_array
    env = gym.make("gym_pusht/PushT-v0", render_mode=render) 
    fixed_state = [0.0, 0.0, 100.0, 200.0, 180.0]  # agent_x,agent_y,block_x,block_y,angle
    observation, info = env.reset(options={"reset_to_state": fixed_state})
    r = 0
    #print(individual)
    for i in range(len(individual) // 2):
        action = individual[2 * i : 2 * i + 2]
        observation, reward, terminated, truncated, info = env.step(action)
        #input()
        #print(f"action:{action}")
        #print(f"observation:{observation}")
        #print(f"reward:{reward}")
        image = env.render()
        r = reward
        if terminated or truncated:
            observation, info = env.reset(options={"reset_to_state": fixed_state})   

    env.close()

    return r


# -------------------------------------------------------------
# Main algorithm
# -------------------------------------------------------------
def main():
    # ---------------------------------------------------------
    # User parameters
    # ---------------------------------------------------------
    final_size = 20000        # must be even
    num_lists = 1000           # how many lists per step
    min_val = 0.0
    max_val = 512.0

    # Start with an empty list
    best_list_so_far = []

    print("\nStarting algorithm...\n")

    # ---------------------------------------------------------
    # Keep adding pairs until list reaches final_size
    # ---------------------------------------------------------
    while len(best_list_so_far) < final_size:
        """
        print("--------------------------------------")
        print("Currently filled elements:", best_list_so_far)
        print("Step: adding the next pair...")
        print("--------------------------------------")
        """

        best_reward = None
        best_candidate = None

        # -----------------------------------------------------
        # Generate several individuals with random last two values
        # -----------------------------------------------------
        for i in range(num_lists):

            # Copy the best list so far
            candidate = best_list_so_far.copy()

            # Add two new random elements
            a = random.uniform(min_val, max_val)
            b = random.uniform(min_val, max_val)
            candidate.append(a)
            candidate.append(b)

            # Evaluate fitness (sum of all elements)
            if(i == final_size - 1):
                reward = fitness_function(candidate, render="human")
            else:
                reward = fitness_function(candidate)
            # print(reward)

            # Keep the best one
            if (best_reward is None) or (reward > best_reward):
                best_reward = reward
                best_candidate = candidate
                chosen_pair = (a, b)

        # After checking all candidates, accept the best pair
        best_list_so_far = best_candidate

        # print("Chosen pair:", chosen_pair)
        print("Best reward :", best_reward)
        print()

    # ---------------------------------------------------------
    # Finished
    # ---------------------------------------------------------
    print("=====================================")
    print("Final list:", best_list_so_far)
    print("Final sum (reward):", fitness_function(best_list_so_far))
    print("=====================================")

# -------------------------------------------------------------
# Run program
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
