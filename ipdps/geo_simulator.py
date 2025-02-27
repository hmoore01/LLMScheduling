import random
import copy
import numpy as np
import math
# import matplotlib.pyplot as plt
import time
import pickle
import math
import argparse
import os
import csv
# import Env



random.seed(10)
# np.set_printoptions(precision=5)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workload', type=str, help='workload type', default='azure', choices=['azure', 'huawei'])
    parser.add_argument('-l', '--laxity', type=int, help='deadline laxity', default=10)
    parser.add_argument('-s', '--slo', type=float, help='slo contraint', default=0.05)
    parser.add_argument('-e', '--epoch', type=int, help='number of epoch', default=96)
    parser.add_argument('-t', '--time', type=int, help='decision time', default=180)
    parser.add_argument('-n', '--node', type=int, help='number of nodes', default=4)
    parser.add_argument('-d', '--duration', type=int, help='times of duration', default=1)
    parser.add_argument('-r', '--request', type=int, help='times of request', default=20000)
    parser.add_argument('-o', '--origin', type=str, help='origin pattern of request', default='change', 
                        choices=['even', 'coast', 'centre', 'change'])
    parser.add_argument('-f', '--framework', type=str, help='framework', default='even', choices= 
                        ['even', 'coast', 'centre', 'change',
                         'hybrid_v1', 'hybrid_v2', 'hybrid_v3', 'hybrid_v4', 'hybrid_v5',
                         'kimchi', 'greenc'])
    args = parser.parse_args()

    workload = args.workload
    ddl_laxity = args.laxity
    slo_constraint = args.slo
    number_of_epoch = args.epoch
    decision_time = args.time
    number_of_node = args.node
    time_of_duration = args.duration
    time_of_request = args.request
    framework = args.framework
    origin_pattern = args.origin
        
    print(number_of_node, time_of_duration, time_of_request, framework)
    
    a_week_func_distribution_list =[] #(func_id, total_rate, rate_in_900sec)
    if workload == 'azure':
        total_number_of_func_id = 424
        distribution_file="a_week_func_distribution_list_original"
        lut_file="a_week_func_runtime_list_original"
    else:
        total_number_of_func_id =200
        distribution_file="huawei_func_distribution_list_original"
        lut_file="huawei_func_runtime_list_original"
    total_func_list=list(range(total_number_of_func_id))

    with open(distribution_file, "rb") as f:   #Unpickling
        a_week_func_distribution_list = pickle.load(f)
    for epoch in range(96):
        request_intensity=0
        for func_id in range(total_number_of_func_id):
            request_intensity+=int(a_week_func_distribution_list[epoch][func_id,0])
    #     print(epoch, request_intensity*20000)
    # exit()

    print("distribution load success")

    lut_list = []
    with open(lut_file, "rb") as f:   #Unpickling
        lut_list = pickle.load(f)
    lut_arr = np.array(lut_list, dtype=int)
    # print(lut_arr.shape)
    # exit()
    print("runtime load success")
    # print(lut_arr)
    
    objective_arr = np.zeros((number_of_epoch,4))
    leftover_resource_time_arr = np.zeros((number_of_node, 900))
    req_distribution_arr = np.zeros((total_number_of_func_id, number_of_node+1)) ## (number of container per ID, location 0, ..., location n) as one line, 424 lines represent 424 IDs in Azure trace

    time_limit = 60
    regular_vm_list = []
    for i in range(total_number_of_func_id):
        if lut_list[i] > time_limit:
            regular_vm_list.append(i)

    if 'hybrid' not in framework:
        print('epoch', 'total workload', '[slo_rate, cost, carbon, water]')
    else:
        print('epoch', 'phv', 'time')
    pareto_front_value_arr=np.zeros((20,4))
    # for i in range(672):
    for epoch in range(0,number_of_epoch): 
        if origin_pattern == 'even':
            original_req_distribution_arr = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        if origin_pattern == 'coast':
            original_req_distribution_arr = np.array([0.4, 0.1, 0.1, 0.4], dtype=float)
        if origin_pattern == 'centre':
            original_req_distribution_arr = np.array([0.1, 0.4, 0.4, 0.1], dtype=float)
        if origin_pattern == 'change':
            original_req_distribution_arr = np.array([0.1, 0.1, 0.1, 0.1], dtype=float)
            a,b = divmod(epoch, 32)
            original_req_distribution_arr[a] = 0.7 - 0.01875*b
            original_req_distribution_arr[a+1] = 0.1 + 0.01875*b
    # for i in range(11,12):
        s_time = time.time()
        previous_func_type = [] 
        if framework == 'change':
            from Change_Distribution import *
            new_req_distribution_arr = change_distribution(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[epoch], leftover_resource_time_arr, req_distribution_arr, original_req_distribution_arr)
        if framework == 'even':
            from Even_Distribution import *
            new_req_distribution_arr = even_distribution(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[epoch], leftover_resource_time_arr, req_distribution_arr)
        if framework == 'coast':
            from Coast_Distribution import *
            new_req_distribution_arr = coast_distribution(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[epoch], leftover_resource_time_arr, req_distribution_arr)
        if framework == 'centre':
            from Centre_Distribution import *
            new_req_distribution_arr = centre_distribution(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[epoch], leftover_resource_time_arr, req_distribution_arr)
        if framework == 'kimchi':
            from Kimchi import *
            new_req_distribution_arr = kimchi(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[epoch], leftover_resource_time_arr, req_distribution_arr, epoch, original_req_distribution_arr)
        if framework == 'greenc':
            from Greencourier import *
            new_req_distribution_arr = greencourier(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[epoch], leftover_resource_time_arr, req_distribution_arr, epoch, original_req_distribution_arr)
        if framework == 'hybrid_v1':
            from Hybrid_Search_V1 import *
            start_time=time.time()
            pareto_front_design, pareto_front_value, standard_objective = hybrid_search(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[epoch], leftover_resource_time_arr, req_distribution_arr, epoch, original_req_distribution_arr)
            for j in range(20):
                pareto_front_value_arr[j,0]+=pareto_front_value[j][0]
                pareto_front_value_arr[j,1]+=pareto_front_value[j][1]
                pareto_front_value_arr[j,2]+=pareto_front_value[j][2]
                pareto_front_value_arr[j,3]+=pareto_front_value[j][3]
            # print(epoch, time.time()-start_time)
        if framework == 'hybrid_v3':
            from Hybrid_Search_V3 import *
            start_time=time.time()
            pareto_front_design, pareto_front_value, standard_objective = hybrid_search_p(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[epoch], leftover_resource_time_arr, req_distribution_arr, epoch, original_req_distribution_arr)
            for j in range(20):
                pareto_front_value_arr[j,0]+=pareto_front_value[j][0]
                pareto_front_value_arr[j,1]+=pareto_front_value[j][1]
                pareto_front_value_arr[j,2]+=pareto_front_value[j][2]
                pareto_front_value_arr[j,3]+=pareto_front_value[j][3]
            # print(epoch, time.time()-start_time)
        if framework == 'hybrid_v4':
            from Hybrid_Search_V4 import *
            start_time=time.time()
            pareto_front_design, pareto_front_value, standard_objective = hybrid_search_p(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[epoch], leftover_resource_time_arr, req_distribution_arr, epoch, original_req_distribution_arr)
            for j in range(20):
                pareto_front_value_arr[j,0]+=pareto_front_value[j][0]
                pareto_front_value_arr[j,1]+=pareto_front_value[j][1]
                pareto_front_value_arr[j,2]+=pareto_front_value[j][2]
                pareto_front_value_arr[j,3]+=pareto_front_value[j][3]
            # print(epoch, time.time()-start_time)
        
        if framework == 'hybrid_v5':
            from Hybrid_Search_V5 import *
            start_time=time.time()
            pareto_front_design, pareto_front_value, standard_objective = hybrid_search_p(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[epoch], leftover_resource_time_arr, req_distribution_arr, epoch, original_req_distribution_arr)
            for j in range(20):
                pareto_front_value_arr[j,0]+=pareto_front_value[j][0]
                pareto_front_value_arr[j,1]+=pareto_front_value[j][1]
                pareto_front_value_arr[j,2]+=pareto_front_value[j][2]
                pareto_front_value_arr[j,3]+=pareto_front_value[j][3]
            # print(epoch, time.time()-start_time)

        if 'hybrid' not in framework:
            epoch_objectives, leftover_resource_time_arr = simulation(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[epoch], leftover_resource_time_arr, 
                                                                    original_req_distribution_arr, req_distribution_arr, new_req_distribution_arr, epoch)
            print(epoch, np.sum(a_week_func_distribution_list[epoch][:,0]),epoch_objectives)
            objective_arr[epoch] = epoch_objectives
        else:
            phv_value = hvwfg.wfg(np.array(pareto_front_value).astype('double'), np.array(standard_objective).astype('double'))
            print(epoch, phv_value, time.time()-start_time)


    print("utility (M$), cost (k$), carbon (Ton), water (m^3)")
    if 'hybrid' not in framework:
        ave_violation_rate = np.sum(objective_arr[:,0])
        cumulative_cost = np.sum(objective_arr[:,1])
        cumulative_carbon = np.sum(objective_arr[:,2])
        cumulative_water = np.sum(objective_arr[:,3])
        print(ave_violation_rate, cumulative_cost, cumulative_carbon, cumulative_water)
    else:
        for j in range(20):
            ave_violation_rate = pareto_front_value_arr[j,0]
            cumulative_cost = pareto_front_value_arr[j,1]
            cumulative_carbon = pareto_front_value_arr[j,2]
            cumulative_water = pareto_front_value_arr[j,3]
            print(ave_violation_rate, cumulative_cost, cumulative_carbon, cumulative_water)

    # with open('outputs/'+framework+'_l'+str(ddl_laxity)+'_n'+str(number_of_node)+'_d'+str(time_of_duration)+'_r'+str(time_of_request)+'_e'+str(number_of_epoch)+'_t'+str(decision_time)+'_s'+str(slo_constraint*100), 'w') as f:
    #         f.writelines(str(ave_violation_rate) + ',' + str(cumulative_cost) + ',' + str(cumulative_carbon) + ',' + str(cumulative_water))
    # exit()

    # if framework == 'Qtrain':
    #     with open("q_table_n"+str(number_of_node), "wb") as f:   #Pickling
    #         pickle.dump(Q_table, f)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if 'hybrid' not in framework:
        with open('outputs/'+workload+'_'+framework+'_n'+str(number_of_node)+'_d'+str(time_of_duration)+'_r'+str(time_of_request)+'_e'+str(number_of_epoch)+'_'+origin_pattern, 'w') as f:
            f.writelines(str(ave_violation_rate) + ',' + str(cumulative_cost) + ',' + str(cumulative_carbon) + ',' + str(cumulative_water))
    else:
        with open('outputs/'+workload+'_'+framework+'_n'+str(number_of_node)+'_d'+str(time_of_duration)+'_r'+str(time_of_request)+'_e'+str(number_of_epoch)+'_'+origin_pattern, 'w') as f:
            for j in range(20):
                ave_violation_rate = pareto_front_value_arr[j,0]
                cumulative_cost = pareto_front_value_arr[j,1]
                cumulative_carbon = pareto_front_value_arr[j,2]
                cumulative_water = pareto_front_value_arr[j,3]
                f.writelines(str(ave_violation_rate) + ',' + str(cumulative_cost) + ',' + str(cumulative_carbon) + ',' + str(cumulative_water) + '\n')

        
    
    