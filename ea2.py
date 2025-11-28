import gymnasium as gym
import gym_pusht
import random
import time

# -------------------------------------------------------------
# Fitness function
# -------------------------------------------------------------
def fitness_function(individual):
    # Evaluate the sequence of actions in the PushT environment
    env = gym.make("gym_pusht/PushT-v0", render_mode="human")  # human

    fixed_state = [0.0, 0.0, 100.0, 200.0, 180.0]  # agent_x, agent_y, block_x, block_y, angle
    observation, info = env.reset(options={"reset_to_state": fixed_state})
    r = 0.0

    # Each action consists of two consecutive values in the individual
    for i in range(len(individual) // 2):
        action = individual[2 * i : 2 * i + 2]
        observation, reward, terminated, truncated, info = env.step(action)
        r = reward  # keep last reward (or use any other aggregation you like)

        if terminated or truncated:
            observation, info = env.reset(options={"reset_to_state": fixed_state})
        time.sleep(0.2)

    env.close()
    return r


# -------------------------------------------------------------
# Main algorithm
# -------------------------------------------------------------
def main():
    # ---------------------------------------------------------
    # User parameters
    # ---------------------------------------------------------
    final_size = 20          # must be even, since 2 values = 1 action
    num_lists = 10       # how many candidate extensions per step
    min_val = 0.0
    max_val = 512.0

    print("\nStarting algorithm...\n")

    # ---------------------------------------------------------
    # 1) Initialize with the best ONE-ACTION individual
    #    (so we do NOT start from the empty list)
    # ---------------------------------------------------------
    print("Initializing first action (one pair)...\n")

    best_list_so_far = None
    best_reward_so_far = None
    chosen_pair = None

    for _ in range(num_lists):
        a = random.uniform(min_val, max_val)
        b = random.uniform(min_val, max_val)
        candidate = [a, b]

        reward = fitness_function(candidate)

        if (best_reward_so_far is None) or (reward > best_reward_so_far):
            best_reward_so_far = reward
            best_list_so_far = candidate
            chosen_pair = (a, b)

    print("Initial chosen pair :", chosen_pair)
    print("Initial best reward :", best_reward_so_far)
    print("Initial individual  :", best_list_so_far)
    print()

    # ---------------------------------------------------------
    # 2) Now keep extending by ONE ACTION at a time,
    #    but ONLY if the extension is better than the original
    # ---------------------------------------------------------
    while len(best_list_so_far) < final_size:

        print("------------------------------------------------")
        print("Current sequence length:", len(best_list_so_far))
        print("Current best individual:", best_list_so_far)
        print("Current best reward   :", best_reward_so_far)
        print("Trying to add the NEXT action (pair of values)...")
        print("------------------------------------------------")

        best_candidate = None
        best_candidate_reward = None
        chosen_pair = None

        # Generate several individuals by adding ONE new action
        for _ in range(num_lists):

            candidate = best_list_so_far.copy()

            # Add one new random action (two values)
            a = random.uniform(min_val, max_val)
            b = random.uniform(min_val, max_val)
            candidate.append(a)
            candidate.append(b)

            reward = fitness_function(candidate)

            if (best_candidate_reward is None) or (reward > best_candidate_reward):
                best_candidate_reward = reward
                best_candidate = candidate
                chosen_pair = (a, b)

        # Decide whether to accept the extension
        if best_candidate_reward is not None and best_candidate_reward > best_reward_so_far:
            # The best extended individual is better than the current one:
            # accept it and increase the sequence length by one action
            print("Improvement found!")
            print("Chosen pair        :", chosen_pair)
            print("Old best reward    :", best_reward_so_far)
            print("New best reward    :", best_candidate_reward)
            print()

            best_list_so_far = best_candidate
            best_reward_so_far = best_candidate_reward
        else:
            # No extended individual beat the current one:
            # keep the original and do NOT increase the length
            print("No better extension found in this batch.")
            print("Keeping current individual with reward:", best_reward_so_far)
            print("Sequence length stays at:", len(best_list_so_far))
            print()

        # While-loop continues until len(best_list_so_far) == final_size

    # ---------------------------------------------------------
    # Finished
    # ---------------------------------------------------------
    print("=====================================")
    print("Final list  :", best_list_so_far)
    print("Final reward:", fitness_function(best_list_so_far))
    print("Final length:", len(best_list_so_far))
    print("=====================================")


# -------------------------------------------------------------
# Run program
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
