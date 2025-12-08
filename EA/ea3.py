import gymnasium as gym
import gym_pusht
import random
import time
# -------------------------------------------------------------
# Fitness function：用來評分一串動作的好壞（使用同一個 env）
# -------------------------------------------------------------
def fitness_function(individual, env, debug_render=False):
    # 固定機器人和方塊的初始位置
    fixed_state = [20.0, 250, 100.0, 200.0, 0.0]

    # 重設環境到固定位置
    observation, info = env.reset(options={"reset_to_state": fixed_state})

    r = 0.0   # 用來記錄最後一個 action 的 reward

    # individual 是一串數字，每 2 個數字代表 1 個 action
    for i in range(len(individual) // 2):

        action = individual[2 * i : 2 * i + 2]

        # 把 action 丟給環境執行
        observation, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        
        r = reward
        
        # 如果遊戲結束或出錯，就直接結束這次評分
        if terminated or truncated:
            break

        if debug_render:
            time.sleep(0.3)

    return r


# -------------------------------------------------------------
# Main algorithm：主要演算法（一直找更好的動作序列）
# -------------------------------------------------------------
def main():

    # final_size：最後要做到多長的 action 序列（要是偶數，因為 2 個值＝1 個 action）
    final_size = 100

    # 每次要增加的 action 數量（2 個值＝1 個 action）
    action_chunk_size = 5

    # num_lists：每次要產生幾個候選序列來比較
    num_lists = 20

    # 隨機動作的數值範圍（PushT 裡 action 是 0~512）
    min_val = 0.0
    max_val = 512.0

    print("\nStarting algorithm...\n")

    # 在外面建立一次環境，不開 render（加速）
    env = gym.make("gym_pusht/PushT-v0")

    # ---------------------------------------------------------
    # 第一步：先找出「最好的初始序列」
    # ---------------------------------------------------------
    print("Initializing first block of actions...\n")

    best_list_so_far = None
    best_reward_so_far = None
    chosen_pair = None

    for _ in range(num_lists):
        candidate = []
        for _ in range(action_chunk_size):
            a = random.uniform(min_val, max_val)
            b = random.uniform(min_val, max_val)
            candidate.append(a)
            candidate.append(b)

        reward = fitness_function(candidate, env)

        if (best_reward_so_far is None) or (reward > best_reward_so_far):
            best_reward_so_far = reward
            best_list_so_far = candidate
            chosen_pair = candidate

    print("Initial chosen block :", chosen_pair)
    print("Initial best reward  :", best_reward_so_far)
    print("Initial individual   :", best_list_so_far)
    print()

    # ---------------------------------------------------------
    # 第二步：不停嘗試在後面加入新的 action block
    # 但「只有變更好」才採用
    # ---------------------------------------------------------
    while len(best_list_so_far) < final_size:
        """
        print("------------------------------------------------")
        print("Current sequence length:", len(best_list_so_far))
        print("Current best reward   :", best_reward_so_far)
        print("Trying to add the NEXT block of actions...")
        print("------------------------------------------------")
        """

        best_candidate = None
        best_candidate_reward = None
        chosen_pair = None

        for _ in range(num_lists):

            candidate = best_list_so_far.copy()

            for _ in range(action_chunk_size):
                a = random.uniform(min_val, max_val)
                b = random.uniform(min_val, max_val)
                candidate.append(a)
                candidate.append(b)

            reward = fitness_function(candidate, env)

            if (best_candidate_reward is None) or (reward > best_candidate_reward):
                best_candidate_reward = reward
                best_candidate = candidate
                chosen_pair = candidate

        if best_candidate_reward is not None and best_candidate_reward > best_reward_so_far:
            print("Improvement found!")
            print("Old best reward:", best_reward_so_far)
            print("New best reward:", best_candidate_reward)
            
            best_list_so_far = best_candidate
            best_reward_so_far = best_candidate_reward

            print("------------------------------------------------")
            print("Current sequence length:", len(best_list_so_far))
            print("Current best reward   :", best_reward_so_far)
            print("Trying to add the NEXT block of actions...")
            print("------------------------------------------------")
            print()
        else:
            #print("No better extension found in this batch.")
            #print()
            pass

    env.close()

    env = gym.make("gym_pusht/PushT-v0", render_mode="human")
    input("Press Enter to see the final result...")
    print("=====================================")
    print("Final list  :", best_list_so_far)
    print("Final reward:", fitness_function(best_list_so_far, env, debug_render=True))
    print("Final length:", len(best_list_so_far))
    print("=====================================")

    env.close()


if __name__ == "__main__":
    main()
