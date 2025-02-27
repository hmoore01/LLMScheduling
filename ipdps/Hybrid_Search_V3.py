from Env import *
from multiprocessing import Pool
import time
from sklearn.preprocessing import MinMaxScaler
global_min_weight = 0.0005
# scaler = MinMaxScaler((0, 100))
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
    for search_epoch in range(2):
    # for search_epoch in range(100):
        
        search_location = random.choice(list(range(20)))
        print(search_epoch, search_location)
        searched_objectives, searched_req_distribution_arr = perturb_and_simulate_10time(search_location, value_arr, design_arr, weight_arr,
                                                                                         step_size, lut_list, func_distribution_arr, leftover_resource_time_arr, 
                                                                                         original_req_distribution_arr, req_distribution_arr, epoch)
        value_arr[search_location] = searched_objectives
        design_arr[search_location] = searched_req_distribution_arr

    return design_arr, value_arr  ##should return pareto front

def hybrid_search_p(lut_list, func_distribution_arr, leftover_resource_time_arr, req_distribution_arr, epoch, original_req_distribution_arr, step_size=0.05):
    start_time = time.time()
    total_number_of_func_type = func_distribution_arr.shape[0]
    ## loading weights
    number_of_population = 20
    number_of_objective = 4
    number_of_worker = 4
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
    
    standard_objective = copy.deepcopy(new_objectives)
    print(standard_objective)
    exit()
    # standard_objective[0]=1
    
    #normalization part
    min_objs=[element * 0.5 for element in standard_objective]
    max_objs=[element * 1.5 for element in standard_objective]
    scaler = MinMaxScaler((0, 100))
    reference_point=[100]*4
    print(scaler.fit([min_objs, max_objs]))

    value_arr = [new_objectives] * 20
    design_arr = [new_req_distribution_arr] * 20
    update_count=0
    search_epoch_time = time.time()
    for search_epoch in range(50):
        
        ## local search part
        search_location_list = random.sample(list(range(20)), number_of_worker)
        # print(search_epoch, search_location_list, time.time()-search_epoch_time)
        search_epoch_time = time.time()
        arg = []
        output = []
        for search_location in search_location_list:
            arg.append((search_location, value_arr, design_arr, weight_arr, 
                        step_size, lut_list, func_distribution_arr, leftover_resource_time_arr, 
                        original_req_distribution_arr, req_distribution_arr, epoch, scaler))
        
        with Pool(processes=number_of_worker) as pool:
            results = pool.map(perturb_and_simulate_10time_p, arg)

        for result,search_location in zip(results,search_location_list):
            value_arr[search_location], design_arr[search_location] = result
        
        ## GA part
        # print('GA in')
        member_list = search_location_list
        non_member_list = list(range(20))
        for i in sorted(member_list, reverse = True):
            non_member_list.pop(i)
        length_of_member_list = len(member_list)
        all_children_design_list = []
        all_children_value_list = []
        for i in range(length_of_member_list):
            parent_1 = member_list.pop()
            parent_2 = random.sample(non_member_list, k = 1)[0]
            
            parent_design_1 = design_arr[parent_1]
            parent_design_2 = design_arr[parent_2]
            
            child_design_candidate_list = GA(parent_design_1, parent_design_2, step_size)
            all_children_design_list.extend(child_design_candidate_list)
        
        arg = []
        output = []
        for child_design in all_children_design_list:
            arg.append((lut_list, func_distribution_arr, leftover_resource_time_arr, 
                        original_req_distribution_arr, req_distribution_arr, child_design, 
                        epoch))
        
        with Pool(processes=len(all_children_design_list)) as pool:
            results = pool.map(parallel_simulation, arg)

        for result, child_design in zip(results, all_children_design_list):
            # child_value, child_design = result
            child_value, _ = result
            all_children_value_list.append(child_value)
        
        design_arr, value_arr = update_population(value_arr, design_arr, all_children_value_list, all_children_design_list, weight_arr, scaler)
        # print('GA out')    
        
        pareto_front=copy.deepcopy(value_arr)
        for j in range(20):
            pareto_front[j][0]=100-pareto_front[j][0]
        phv_value = hvwfg.wfg(np.array(scaler.transform(pareto_front)).astype('double'), np.array(scaler.transform([reference_point])[0]).astype('double'))
        print(search_epoch, phv_value, time.time()-start_time)
    return design_arr, value_arr, standard_objective  ##should return pareto front

def update_population(value_arr, design_arr, all_children_value_list, all_children_design_list, weight_arr, scaler):
    update_count = 0
    new_design_arr = copy.deepcopy(design_arr)
    new_value_arr = copy.deepcopy(value_arr)
    # mating_indices = list(range(self.population_size))
    update_list = list(range(len(weight_arr)))
    random.shuffle(update_list)
    
    updated = False
    length_of_update_list = len(update_list)
    child_list = list(range(len(all_children_value_list)))
    random.shuffle(child_list)
    
    while child_list:
        index = child_list.pop()
        child_design = all_children_design_list[index]
        child_value = all_children_value_list[index]
        j = 0
        while j < length_of_update_list:
        # for j in update_list:
            the_index = update_list[j]
            candidate = value_arr[the_index]
            weight = weight_arr[the_index]
            replace = False
            # fit_new = self.chebyshev(child_value, weight) #the most promising optimization direction
            # fit_old = self.chebyshev(candidate, weight) #the most promising optimization direction
            fit_new = local_fit(child_value, weight, scaler) #the most promising optimization direction
            fit_old = local_fit(candidate, weight, scaler) #the most promising optimization direction
            if fit_new < fit_old:
                # print('update solution:', fit_new, fit_old, child_design_value, candidate)
                replace = True
            if replace:
                # print('GA update success')
                # print(child_design.shape)
                new_design_arr[the_index] = copy.deepcopy(child_design)
                new_value_arr[the_index] = copy.deepcopy(child_value)
                update_count += 1
                updated = True
                j = length_of_update_list
            else:
                j+=1

    return new_design_arr, new_value_arr
# def local_fit(normalized_values, weight_vector):
#     temp = []
#     for i in range(len(weight_vector)):
#         temp.append((normalized_values[i]) * max(weight_vector[i], global_min_weight)) #(difference * weight)
#     fit = sum(temp)
#     fit += normalized_values[0] * weight_vector[0]
#     return fit

def GA(parent_design_1, parent_design_2, step_size=0.1):
    mutation_rate = 0.5
    child_design_1, child_design_2 = crossover(parent_design_1, parent_design_2)
    # print(parent_design_1[0,:])
    # print(parent_design_2[0,:])
    # print(child_design_1[0,:])
    # print(child_design_2[0,:])
    # exit()
    #if random.uniform(0,1) < mutation_rate:
    child_design_3 = perturb(child_design_1, step_size)
    child_design_4 = perturb(child_design_2, step_size)
    return [child_design_1, child_design_2, child_design_3, child_design_4]

def crossover(parent_design_1, parent_design_2):
    child_1 = copy.deepcopy(parent_design_1)
    child_2 = copy.deepcopy(parent_design_2)
    number_of_columns = child_1.shape[1]
    # number_of_columns = self.number_of_city
    avail_column = list(range(number_of_columns))
    number_of_columns_to_be_exchanged = random.choices(list(range(1,number_of_columns)), k =1)[0]
    #exchange_list = random.sample(list(range(number_of_columns)), k = number_of_columns_to_be_exchanged)
    exchange_list = random.sample(avail_column, k = number_of_columns_to_be_exchanged)
    # print('cross over index:', exchange_list)
    
    # crossover
    for i in range(number_of_columns):
        if i in exchange_list:
            child_1[:,i] = copy.deepcopy(parent_design_2[:,i])
            child_2[:,i] = copy.deepcopy(parent_design_1[:,i])
        else:
            child_1[:,i] = copy.deepcopy(parent_design_1[:,i])
            child_2[:,i] = copy.deepcopy(parent_design_2[:,i])
    
    sum_1 = sum(child_1[0,1:])
    sum_2 = sum(child_2[0,1:])
    child_1[:,1:]=child_1[:,1:]/sum_1
    child_2[:,1:]=child_2[:,1:]/sum_2

    return (child_1,child_2)

def perturb_and_simulate_10time_p(args):
    search_location, value_arr, design_arr, weight_arr, step_size, lut_list, func_distribution_arr, leftover_resource_time_arr, original_req_distribution_arr, req_distribution_arr, epoch, scaler = args
    return perturb_and_simulate_10time(search_location, value_arr, design_arr, weight_arr, 
                                       step_size, lut_list, func_distribution_arr, leftover_resource_time_arr, 
                                       original_req_distribution_arr, req_distribution_arr, epoch, scaler)

def perturb_and_simulate_10time(search_location, value_arr, design_arr, weight_arr,
                                step_size, lut_list, func_distribution_arr, leftover_resource_time_arr,
                                original_req_distribution_arr, req_distribution_arr, epoch, scaler):
    
    the_value = copy.deepcopy(value_arr[search_location])
    the_design = copy.deepcopy(design_arr[search_location])
    for _ in range(5):
        search_req_distribution_arr = perturb(the_design, step_size)
        search_objectives, search_leftover_resource_time_arr = simulation(lut_list, func_distribution_arr, leftover_resource_time_arr, 
                                                                        original_req_distribution_arr, req_distribution_arr, search_req_distribution_arr, epoch)
        if local_fit(search_objectives, weight_arr[search_location], scaler) < local_fit(the_value, weight_arr[search_location], scaler):
            the_value = copy.deepcopy(search_objectives)
            the_design = copy.deepcopy(search_req_distribution_arr)
    
    return the_value, the_design

# def local_fit(normalized_values, weight_vector, scaler):
#     fit_0 = (1.64-normalized_values[0])*max(weight_vector[0], global_min_weight)*1000
#     fit_1 = normalized_values[1]*max(weight_vector[1], global_min_weight)
#     fit_2 = normalized_values[2]*max(weight_vector[2], global_min_weight)*100
#     if len(weight_vector) > 3:
#         fit_3 = normalized_values[3]*max(weight_vector[3], global_min_weight)*20
#     else:
#         fit_3 = 0
#     fit = max(fit_0, fit_1, fit_2, fit_3)
#     # print(normalized_values)
#     return fit

def local_fit(real_values, weight_vector, scaler):
    normalized_values = scaler.transform([real_values])[0]
    fit_0 = (100 - normalized_values[0])*max(weight_vector[1], global_min_weight)
    # fit_0 = normalized_values[0]*max(weight_vector[1], global_min_weight)
    fit_1 = normalized_values[1]*max(weight_vector[1], global_min_weight)
    fit_2 = normalized_values[2]*max(weight_vector[2], global_min_weight)
    if len(weight_vector) > 3:
        fit_3 = normalized_values[3]*max(weight_vector[3], global_min_weight)
    else:
        fit_3 = 0
    fit = max(fit_0, fit_1, fit_2, fit_3)
    fit = sum([fit_0, fit_1, fit_2, fit_3])
    # print(normalized_values)
    return fit

def perturb(new_req_distribution_arr, step_size):
    # print(new_req_distribution_arr)
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

def parallel_simulation(args):
    lut_list, func_distribution_arr, leftover_resource_time_arr, original_req_distribution_arr, req_distribution_arr, candidate_req_distribution, epoch = args
    return simulation(lut_list, func_distribution_arr, leftover_resource_time_arr, original_req_distribution_arr, req_distribution_arr, candidate_req_distribution, epoch)
