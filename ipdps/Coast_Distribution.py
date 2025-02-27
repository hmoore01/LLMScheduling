from Env import *

def coast_distribution(lut_list, func_distribution_arr, leftover_resource_time_arr, req_distribution_arr):
    total_number_of_func_type = func_distribution_arr.shape[0]

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
    pass

    epoch_number_of_func_type = len(scored_prewarm_func_list)

    new_req_distribution_arr = np.ones((epoch_number_of_func_type, 4+1))
    new_req_distribution_arr[:,1]=0.4
    new_req_distribution_arr[:,2]=0.1
    new_req_distribution_arr[:,3]=0.1
    new_req_distribution_arr[:,4]=0.4
    for i in range(epoch_number_of_func_type):
        new_req_distribution_arr[i,0]=scored_prewarm_func_list[i]
    
    return new_req_distribution_arr