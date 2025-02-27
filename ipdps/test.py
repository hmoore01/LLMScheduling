import numpy as np
import pickle

# def generate_first_strategy_candidates(current_distribution, step_size=0.1):
#     candidates = []
#     num_elements = len(current_distribution)

#     # Apply step adjustments to all pairs of elements
#     for i in range(num_elements):
#         for j in range(num_elements):
#             if i != j:
#                 # Candidate with +step_size for element i and -step_size for element j
#                 candidate = np.copy(current_distribution)
#                 candidate[i] += step_size
#                 candidate[j] -= step_size

#                 # Ensure the values remain within valid bounds [0, 1]
#                 candidate = np.clip(candidate, 0, 1)

#                 # Normalize the candidate to sum to 1
#                 candidate /= np.sum(candidate)

#                 # Add to the list of candidates
#                 candidates.append(candidate)

#     return candidates

# epoch_number_of_func_type=20
# new_req_distribution_arr = np.ones((epoch_number_of_func_type, 4), dtype=float)
# new_req_distribution_arr[:,0]=0.25
# new_req_distribution_arr[:,1]=0.25
# new_req_distribution_arr[:,2]=0.25
# new_req_distribution_arr[:,3]=0.25
# # new_req_distribution_arr[:,4]=0.25
# print(new_req_distribution_arr[0])
# print(generate_first_strategy_candidates(new_req_distribution_arr[0]))
# print(len(generate_first_strategy_candidates(new_req_distribution_arr[0])))

with open('outputs/test', 'w') as f:
    for j in range(20):
        f.writelines(str(j)+'\n')