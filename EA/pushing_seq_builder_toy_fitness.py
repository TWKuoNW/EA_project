# Same code, but with an added "toy" fitness function which just adds up all the values. In the end the indivitual with the lowest value wins.

import random

# -------------------------------------------------------------
# Fitness function
# -------------------------------------------------------------
def fitness_function(individual):
    # Toy fitness function:
    # Cost = sum of all elements in the list
    total = 0.0
    for x in individual:
        total += x
    return total

# -------------------------------------------------------------
# Main algorithm
# -------------------------------------------------------------
def main():

    # ---------------------------------------------------------
    # User parameters
    # ---------------------------------------------------------
    final_size = 10          # must be even
    num_lists = 5            # how many lists per step
    min_val = 0.0
    max_val = 512.0

    # Start with an empty list
    best_list_so_far = []

    print("\nStarting algorithm...\n")

    # ---------------------------------------------------------
    # Keep adding pairs until list reaches final_size
    # ---------------------------------------------------------
    while len(best_list_so_far) < final_size:

        print("--------------------------------------")
        print("Currently filled elements:", best_list_so_far)
        print("Step: adding the next pair...")
        print("--------------------------------------")

        best_cost = None
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
            cost = fitness_function(candidate)

            # Keep the best one
            if (best_cost is None) or (cost < best_cost):
                best_cost = cost
                best_candidate = candidate
                chosen_pair = (a, b)

        # After checking all candidates, accept the best pair
        best_list_so_far = best_candidate

        print("Chosen pair:", chosen_pair)
        print("Best cost :", best_cost)
        print()

    # ---------------------------------------------------------
    # Finished
    # ---------------------------------------------------------
    print("=====================================")
    print("Final list:", best_list_so_far)
    print("Final sum (cost):", fitness_function(best_list_so_far))
    print("=====================================")


# -------------------------------------------------------------
# Run program
# -------------------------------------------------------------
if __name__ == "__main__":
    main()