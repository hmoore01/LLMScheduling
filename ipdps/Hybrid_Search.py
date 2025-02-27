from Env import *
global_min_weight = 0.0005
# np.set_printoptions(precision=5)

def hybrid_search(lut_list, func_distribution_arr, leftover_resource_time_arr, req_distribution_arr, epoch, original_req_distribution_arr, step_size=0.1):
    total_number_of_func_type = func_distribution_arr.shape[0]
    ## loading weights
    number_of_population = 20
    number_of_objective = 4
    with open('data/pop'+str(number_of_population)+'_obj'+str(number_of_objective)+'_weight.csv', newline='') as f:
        reader = csv.reader(f)
        weight_arr = list(reader)
    for i in range(len(weight_arr)):
        weight_arr[i] = [float(j) for j in weight_arr[i]]
    # original_leftover_resource_time_arr = copy.deepcopy(leftover_resource_time_arr)

    #record prewarm func id
    prewarm_func_list = []
    func_invocation_list = []
    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id,0]!=0:
            prewarm_func_list.append(func_id)
            func_invocation_list.append(func_distribution_arr[func_id,0])
    
    #record prewarm func id based on scoring
    scored_prewarm_func_list = []
    scored_invocation_count_list = []
    while len(func_invocation_list):
        max_invocation = max(func_invocation_list)
        max_invocation_index = func_invocation_list.index(max_invocation)
        scored_invocation_count_list.append(func_invocation_list.pop(max_invocation_index))
        scored_prewarm_func_list.append(prewarm_func_list.pop(max_invocation_index))

    epoch_number_of_func_type = len(scored_prewarm_func_list)

    new_req_distribution_arr = np.ones((epoch_number_of_func_type, 4+1), dtype=float)
    new_req_distribution_arr[:,1]=0.25
    new_req_distribution_arr[:,2]=0.25
    new_req_distribution_arr[:,3]=0.25
    new_req_distribution_arr[:,4]=0.25
    for i in range(epoch_number_of_func_type):
        new_req_distribution_arr[i,0]=scored_prewarm_func_list[i]

    new_objectives, new_leftover_resource_time_arr = simulation(lut_list, func_distribution_arr, leftover_resource_time_arr, 
                                                                        original_req_distribution_arr, req_distribution_arr, new_req_distribution_arr, epoch)
    
    
    value_arr = [new_objectives] * 20
    design_arr = [new_req_distribution_arr] * 20
    update_count=0
    # for search_epoch in range(2):
    for search_epoch in range(100):
        
        search_location = random.choice(list(range(20)))
        print(search_epoch, search_location)
        for _ in range(10):
            search_req_distribution_arr = perturb(design_arr[search_location], step_size)
            search_objectives, search_leftover_resource_time_arr = simulation(lut_list, func_distribution_arr, leftover_resource_time_arr, 
                                                                            original_req_distribution_arr, req_distribution_arr, search_req_distribution_arr, epoch)
            if local_fit(search_objectives, weight_arr[search_location]) < local_fit(value_arr[search_location], weight_arr[search_location]):
                update_count+=1
                value_arr[search_location] = copy.deepcopy(search_objectives)
                design_arr[search_location] = copy.deepcopy(search_req_distribution_arr)

    return design_arr, value_arr  ##should return pareto front

# def local_fit(normalized_values, weight_vector):
#     temp = []
#     for i in range(len(weight_vector)):
#         temp.append((normalized_values[i]) * max(weight_vector[i], global_min_weight)) #(difference * weight)
#     fit = sum(temp)
#     fit += normalized_values[0] * weight_vector[0]
#     return fit

def local_fit(normalized_values, weight_vector):
    fit_0 = normalized_values[0]*max(weight_vector[0], global_min_weight)*100
    fit_1 = normalized_values[1]*max(weight_vector[1], global_min_weight)
    fit_2 = normalized_values[2]*max(weight_vector[2], global_min_weight)*100
    if len(weight_vector) > 3:
        fit_3 = normalized_values[3]*max(weight_vector[3], global_min_weight)*20
    else:
        fit_3 = 0
    fit = max(fit_0, fit_1, fit_2, fit_3)
    return fit

def perturb(new_req_distribution_arr, step_size):
    search_req_distribution_arr = copy.deepcopy(new_req_distribution_arr)
    number_of_row, number_of_column = search_req_distribution_arr.shape
    avail_column = list(range(1, number_of_column))
    avail_row = list(range(number_of_row))
    # avail_row = list(range(5))

    perturb_flag = False
    while True:
        perturb_column_list = random.sample(avail_column, 2)
        # perturb_row = random.sample(avail_row, 1)[0]
        perturb_row = avail_row

        if np.all(search_req_distribution_arr[perturb_row,perturb_column_list[0]] > step_size):
            search_req_distribution_arr[perturb_row,perturb_column_list[1]]+=step_size
            search_req_distribution_arr[perturb_row,perturb_column_list[0]]-=step_size
            perturb_flag = True
        if perturb_flag:
            break

    # sample_flag = 0
    # while not sample_flag:
    #     perturb_column_list = random.sample(avail_column, 2)
    #     min_value = min(search_req_distribution_arr[perturb_row, perturb_column_list[0]], search_req_distribution_arr[perturb_row, perturb_column_list[1]])
    #     max_value = max(search_req_distribution_arr[perturb_row, perturb_column_list[0]], search_req_distribution_arr[perturb_row, perturb_column_list[1]])
        
    #     if min_value == max_value and max_value == 0:
    #         sample_flag = 0
    #     else:
    #         sample_flag = 1

    # if min_value == 0:
    #     max_perturb_value = max_value
    #     if max_value < step_size:
    #         search_req_distribution_arr[perturb_row, perturb_column_list[0]] = new_req_distribution_arr[perturb_row, perturb_column_list[1]]
    #         search_req_distribution_arr[perturb_row, perturb_column_list[1]] = new_req_distribution_arr[perturb_row, perturb_column_list[0]]
    #     else:
    #         perturb_value = random.uniform(step_size, max_perturb_value)
    #         perturb_value = (perturb_value//step_size) * step_size
    #         if search_req_distribution_arr[perturb_row, perturb_column_list[0]] > search_req_distribution_arr[perturb_row, perturb_column_list[1]]:
    #             search_req_distribution_arr[perturb_row, perturb_column_list[0]] -= perturb_value
    #             search_req_distribution_arr[perturb_row, perturb_column_list[1]] += perturb_value
    #         if search_req_distribution_arr[perturb_row, perturb_column_list[0]] <= search_req_distribution_arr[perturb_row, perturb_column_list[1]]:
    #             search_req_distribution_arr[perturb_row, perturb_column_list[0]] += perturb_value
    #             search_req_distribution_arr[perturb_row, perturb_column_list[1]] -= perturb_value
    # else:
    #     max_perturb_value = min_value
    #     perturb_value = random.uniform(step_size, max_perturb_value)
    #     perturb_value = (perturb_value//step_size) * step_size
    #     if search_req_distribution_arr[perturb_row, perturb_column_list[0]] > search_req_distribution_arr[perturb_row, perturb_column_list[1]]:
    #         search_req_distribution_arr[perturb_row, perturb_column_list[0]] -= perturb_value
    #         search_req_distribution_arr[perturb_row, perturb_column_list[1]] += perturb_value
    
    return search_req_distribution_arr