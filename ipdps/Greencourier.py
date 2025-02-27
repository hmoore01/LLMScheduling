from Env import *
from multiprocessing import Pool


def greencourier(lut_list, func_distribution_arr, leftover_resource_time_arr, req_distribution_arr, epoch, original_req_distribution_arr, max_epochs=10, violation_threshold=0.05, num_workers=4):
    # Prewarming containers based on functions
    total_number_of_func_type = func_distribution_arr.shape[0]

    # record prewarm func id
    prewarm_func_list = []
    func_invocation_list = []
    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id, 0] != 0:
            prewarm_func_list.append(func_id)
            func_invocation_list.append(func_distribution_arr[func_id, 0])

    # record prewarm func id based on scoring
    scored_prewarm_func_list = []
    scored_invocation_count_list = []
    while len(func_invocation_list):
        max_invocation = max(func_invocation_list)
        max_invocation_index = func_invocation_list.index(max_invocation)
        scored_invocation_count_list.append(func_invocation_list.pop(max_invocation_index))
        scored_prewarm_func_list.append(prewarm_func_list.pop(max_invocation_index))
    pass

    epoch_number_of_func_type = len(scored_prewarm_func_list)

    # Initialize best metrics
    best_score = -np.inf
    best_req_distribution = np.ones((epoch_number_of_func_type, 4 + 1)) * 0.25
    for i in range(epoch_number_of_func_type):
        best_req_distribution[i,0]=scored_prewarm_func_list[i]

    # Run the heuristic for a maximum number of epochs
    for _ in range(max_epochs):
        candidate_results = []
        # Generate candidates for the current distribution
        candidate_req_distribution_base = best_req_distribution[i, 1:]  # Exclude function ID (column 0)
        candidates = generate_first_strategy_candidates(candidate_req_distribution_base)

        # Prepare arguments for parallel simulation
        args_list = []
        full_candidate_distributions = []
        for candidate in candidates:
            # Create a copy of best_req_distribution for each candidate
            candidate_req_distribution = np.copy(best_req_distribution)

            # Apply the same candidate to all function types
            for i in range(epoch_number_of_func_type):
                candidate_req_distribution[i, 1:] = candidate  # Apply the candidate to each function type

            # Add the setup candidate to the argument list
            args_list.append(
                (simulation, lut_list, func_distribution_arr, leftover_resource_time_arr,
                    original_req_distribution_arr, req_distribution_arr, candidate_req_distribution, epoch)
            )

            # Save the full candidate distribution for later selection
            full_candidate_distributions.append(candidate_req_distribution)

        # Run simulations in parallel using multiprocessing
        with Pool(processes=num_workers) as pool:
            results = pool.map(parallel_simulation, args_list)

        # Collect results from parallel execution
        for result, full_candidate_distribution in zip(results, full_candidate_distributions):
            new_objectives, new_leftover_resource_time_arr = result

            # Extract key metrics: violation rate, carbon emissions
            violation_rate = new_objectives[0]  # Violation rate (%)
            carbon_emissions = new_objectives[2]  # Carbon emissions (Ton)

            # Calculate score based on violation rate and carbon emissions
            score = calculate_score(violation_rate, carbon_emissions, violation_threshold)

            # Store the candidate results
            candidate_results.append((full_candidate_distribution, score))

        # Select the best candidate based on the highest score
        for candidate_req_distribution, score in candidate_results:
            if score > best_score:
                best_score = score
                best_req_distribution = np.copy(candidate_req_distribution)

    # Return the best new_req_distribution_arr found
    return best_req_distribution


# def calculate_score(violation_rate, carbon_emissions, violation_threshold):

#     if violation_rate > violation_threshold:
#         return -np.inf  # Disqualify if the violation rate exceeds the threshold

#     # Higher score for lower carbon emissions and lower violation rate
#     score = 1000 / (carbon_emissions + 1) - violation_rate  # Example scoring formula
#     return score

def calculate_score(violation_rate, carbon_emissions, violation_threshold):

    # Higher score for lower carbon emissions and high utility
    score = 1 / (carbon_emissions + 1) + violation_rate  # Example scoring formula
    return score

def generate_first_strategy_candidates(current_distribution, step_size=0.1):
    candidates = []
    num_elements = len(current_distribution)

    # Apply step adjustments to all pairs of elements
    for i in range(num_elements):
        for j in range(num_elements):
            if i != j:
                # Candidate with +step_size for element i and -step_size for element j
                candidate = np.copy(current_distribution)
                candidate[i] += step_size
                candidate[j] -= step_size

                # Ensure the values remain within valid bounds [0, 1]
                candidate = np.clip(candidate, 0, 1)

                # Normalize the candidate to sum to 1
                candidate /= np.sum(candidate)

                # Add to the list of candidates
                candidates.append(candidate)

    return candidates

def parallel_simulation(args):
    simulation, lut_list, func_distribution_arr, leftover_resource_time_arr, original_req_distribution_arr, req_distribution_arr, candidate_req_distribution, epoch = args
    return simulation(lut_list, func_distribution_arr, leftover_resource_time_arr, original_req_distribution_arr, req_distribution_arr, candidate_req_distribution, epoch)