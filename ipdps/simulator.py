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
from sklearn.cluster import KMeans 


random.seed(10)
# np.random.seed(10)
# per func type: 2000 
container_startup_time = 10 #sec
container_showdown_time = 10 #sec
container_idle_resource = 100 #100mi CPU
container_startup_resource = 200 #100mi CPU
number_of_node = 8
number_of_A100_node = 4
number_of_H100_node = 4
epoch_length = 900
real_time_node_load_arr = np.zeros((number_of_node, epoch_length))
number_of_core_per_node = 128
number_of_gpu_per_node = 4
max_cpu_time = number_of_node*number_of_core_per_node*1000
this_interval_avail_resource_time = np.ones(epoch_length,dtype=float)*max_cpu_time
next_interval_avail_resource_time = np.ones(epoch_length,dtype=float)*max_cpu_time
resource_per_request = 2
ddl_laxity = 10
# resource = 0.1
# per_node_cooling_efficiency = []
global_min_weight = 0.0005
first_interval_flag = 1

# container_startup_time = 0 #sec
# container_showdown_time = 0 #sec
# container_idle_resource = 0 #100mi CPU
# container_startup_resource = 0 #100mi CPU

vm_startup_time = 0
vm_shutdown_time = 30

idle_res_usage = 2

carbon_density_list = [241.7, 221.5, 210.5, 202.1, 199.4, 199.8, 203.9, 223.1, 
                           232.0, 244.1, 229.6, 228.9, 229.9, 227.0, 229.9, 249.0,
                           246.2, 259.6, 273.2, 280.9, 282.4, 279.3,274.4, 260.4] #gram/kWh

water_density = 0.53 #L/KWh
cop_factor_list = [0.85, 0.92, 1.02, 1.10, 1.14, 1.20, 1.21, 1.22, 
                    1.21, 1.20, 1.15, 1.11, 1.03, 0.96, 0.91, 0.816, 
                    0.83, 0.82, 0.82, 0.81, 0.81, 0.80, 0.80, 0.80]
cop = 1.5
price_list = [10.0, 10.0 ,10.0 ,10.0, 10.0, 10.0, 11.5, 11.5, 
                11.5, 11.5, 11.5, 11.5, 11.5, 18.7, 18.7, 18.7, 
                18.7, 18.7, 11.5, 11.5, 11.5, 11.5, 11.5, 10.0]

def local_fit(normalized_values, weight_vector):
    temp = []
    for i in range(len(weight_vector)):
        temp.append((normalized_values[i]) * max(weight_vector[i], global_min_weight)) #(difference * weight)
    fit = sum(temp)
    fit += normalized_values[0] * weight_vector[0]
    return fit

def local_fit2(normalized_values, weight_vector):
    temp = normalized_values * weight_vector
    fit = sum(temp) + normalized_values[0] * weight_vector[0]
    return fit


def simulation(vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, leftover_resource_time_arr, func_priority_list, lut_list):
    total_number_of_func_type = func_distribution_arr.shape[0]
    slo_violation_arr = np.zeros((total_number_of_func_type, 2))
    last_epoch_func_list = []
    
    scored_prewarm_func_list = func_priority_list
            
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*epoch_length)) ##two times of epoch length means considering left over resource usage for next epoch
    cumulative_resource_time_arr[:,:epoch_length] += leftover_resource_time_arr
    
    #shutdown unnecessary vm based on arrival prediction
    for func_id in range(total_number_of_func_type):
        if func_id not in scored_prewarm_func_list:
            if np.any(vm_distribution_arr[func_id,0] != 0):
                previous_prewarm_location_list = list(np.nonzero(vm_distribution_arr[func_id,1:])[0])
                for node in previous_prewarm_location_list:
                    cumulative_resource_time_arr[node,:vm_shutdown_time]+=idle_res_usage #shutdown

    ##calculate startup latency
    vm_startup_time_list = []
    number_of_container_per_node = np.count_nonzero(new_vm_distribution_arr[:,1:], axis=0)
    # print(number_of_container_per_node)
    for node_id in range(number_of_node):
        number_of_coloation_container = number_of_container_per_node[node_id]
        if node_id < 2:
            vm_startup_time_list.append(2*number_of_coloation_container)  #intra-rack delay
            # vm_startup_time_list.append(0*number_of_coloation_container)  #intra-rack delay
        else:
            vm_startup_time_list.append(5*number_of_coloation_container)
            # vm_startup_time_list.append(0*number_of_coloation_container)  
            
    # print('number of containers:', number_of_container_per_node)

    vm_distribution_arr = copy.deepcopy(new_vm_distribution_arr)
    for node_id in range(number_of_node):
        queued_func_id_list=[]
        queued_func_req_list=[]
        queued_func_delay_list=[]
        load_list = []
        # print(  , prewarm_func_list)
        for second_time in range(epoch_length):
            ##process queued func first
            updated_queued_func_id_list=[]
            updated_queued_func_req_list=[]
            updated_queued_func_delay_list=[]
            for i in range(len(queued_func_id_list)):
                func_id=queued_func_id_list[i]
                number_of_invocation=queued_func_req_list[i]
                delay_time=queued_func_delay_list[i]
                container_location = np.nonzero(vm_distribution_arr[func_id,:])[0][0]
                container_location = node_id
                if second_time >= vm_startup_time_list[container_location]:
                    load_list=[]
                    for node_location in range(number_of_node):
                        load_list.append(max(cumulative_resource_time_arr[node_location,second_time:second_time+15]))
                    resource_time_arr, slo_violation_count, disused_cold_startup, delay_time = calculate_resource_time_v5(load_list, lut_list, func_id, number_of_invocation, node_id, delay_time)
                    cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                    slo_violation_arr[func_id,0]+=slo_violation_count
                    # slo_violation_arr[func_id,1]+=number_of_invocation
                    if disused_cold_startup:
                        if delay_time > 0:
                            updated_queued_func_id_list.append(func_id)
                            updated_queued_func_req_list.append(disused_cold_startup)
                            updated_queued_func_delay_list.append(delay_time)
                        else:
                            print("Warning!!!")
                            exit()
                else:
                    ##vm not warm up yet
                    if func_id not in last_epoch_func_list:
                        updated_queued_func_id_list.append(func_id)
                        updated_queued_func_req_list.append(number_of_invocation)
                        updated_queued_func_delay_list.append(delay_time+1)
                    else:
                        load_list=[]
                        for node_location in range(number_of_node):
                            load_list.append(max(cumulative_resource_time_arr[node_location,second_time:second_time+15]))
                        resource_time_arr, slo_violation_count, disused_cold_startup, delay_time = calculate_resource_time_v5(load_list, lut_list, func_id, number_of_invocation, node_id, delay_time)
                        cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                        slo_violation_arr[func_id,0]+=slo_violation_count
                        # slo_violation_arr[func_id,1]+=number_of_invocation
                        if disused_cold_startup:
                            if delay_time > 0:
                                updated_queued_func_id_list.append(func_id)
                                updated_queued_func_req_list.append(disused_cold_startup)
                                updated_queued_func_delay_list.append(delay_time)
                            else:
                                print("Warning!!!")
            
            ## update queue list
            queued_func_id_list = copy.deepcopy(updated_queued_func_id_list)
            queued_func_req_list = copy.deepcopy(updated_queued_func_req_list)
            queued_func_delay_list = copy.deepcopy(updated_queued_func_delay_list)

            ## process just-come-in func
            for i in range(len(scored_prewarm_func_list)):
                func_id = scored_prewarm_func_list[i]
                number_of_invocation = func_distribution_arr[func_id,second_time+1]
                if vm_distribution_arr[func_id,0] != np.count_nonzero(vm_distribution_arr[func_id,1:]):
                    print("Warning!", func_id, vm_distribution_arr[func_id,:])
                    exit()
                splitted_number_of_invocation = split_request(number_of_invocation, node_id, vm_distribution_arr[func_id,:])
                # if func_id == 24 and splitted_number_of_invocation:
                #     print(splitted_number_of_invocation, node_id, func_id, vm_distribution_arr[func_id,:])
                #     pass
                container_location = node_id
                # print('yes')
                if splitted_number_of_invocation:
                    slo_violation_arr[func_id,1]+=splitted_number_of_invocation
                    if vm_distribution_arr[func_id,container_location+1]:
                        # if func_id==24 and node_id==3:
                        #     print('yes')
                        #     exit()
                        if second_time >= vm_startup_time_list[container_location]:
                            delay_time = 0
                        else: #survived vm from last round
                            if func_id not in last_epoch_func_list:
                                delay_time = 1
                                queued_func_id_list.append(func_id)
                                queued_func_req_list.append(splitted_number_of_invocation)
                                queued_func_delay_list.append(delay_time)
                            else:
                                delay_time = 0
                        if delay_time == 0:
                            load_list=[]
                            for node_location in range(number_of_node):
                                load_list.append(max(cumulative_resource_time_arr[node_location,second_time:second_time+15]))
                            # if func_id ==24 and node_id==3:
                            #     print('yes', load_list[3])
                            resource_time_arr, slo_violation_count, disused_cold_startup, delay_time = calculate_resource_time_v5(load_list, lut_list, func_id, splitted_number_of_invocation, node_id, delay_time)
                            cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                            slo_violation_arr[func_id,0]+=slo_violation_count
                            # slo_violation_arr[func_id,1]+=splitted_number_of_invocation
                            if disused_cold_startup:
                                if delay_time > 0:
                                    queued_func_id_list.append(func_id)
                                    queued_func_req_list.append(disused_cold_startup)
                                    queued_func_delay_list.append(delay_time)
                                else:
                                    print("Warning!!!")
                    else:
                        slo_violation_arr[func_id,0]+=splitted_number_of_invocation
                        # slo_violation_arr[func_id,1]+=splitted_number_of_invocation

        load_list = []
        for i in range(number_of_node):
            load_list.append(max(cumulative_resource_time_arr[i,:])/number_of_core_per_node)
        
        for i in range(len(queued_func_id_list)):
            the_func_id = queued_func_id_list[i]
            slo_violation_count = queued_func_req_list[i]
            slo_violation_arr[the_func_id,0] += slo_violation_count
    
    total_invocation =  np.sum(slo_violation_arr[:,1]) 
    total_violation = np.sum(slo_violation_arr[:,0])
        
    return cumulative_resource_time_arr, int(total_violation), int(total_invocation), vm_distribution_arr, slo_violation_arr

def split_request(number_of_invocation, node_id, vm_distribution_arr):
    number_of_container = vm_distribution_arr[0]
    if number_of_container:
        primary_container_location = np.nonzero(vm_distribution_arr[1:])[0][0]
        if vm_distribution_arr[node_id+1]: 
            if node_id == primary_container_location:
                shared_invocation = number_of_invocation//number_of_container + number_of_invocation%number_of_container
            else:
                shared_invocation = number_of_invocation//number_of_container
        else:
            shared_invocation = 0
    else:
        if node_id == 0:
            shared_invocation = copy.deepcopy(number_of_invocation)
        else:
            shared_invocation = 0
    
    return shared_invocation
   
def calculate_resource_time_v5(nodes_load_list, lut_list, func_id, number_of_invocation, prewarm_location, delay_time): 
    node_load = nodes_load_list[prewarm_location]
    base_execution_time = lut_list[func_id] #baseline execution time
    ddl_time = base_execution_time * ddl_laxity
    resource = resource_per_request #core CPU
    
    resource_time_arr = np.zeros((number_of_node, 900))
    slo_violation_count = 0

    prewarm_shortage = number_of_invocation

    ##check the ddl before execution
    if delay_time + base_execution_time > ddl_time and prewarm_shortage:
        delay_time=-1
        slo_violation_count = number_of_invocation
        disused_cold_startup = 0       
        prewarm_shortage = 0

    if prewarm_shortage:
        if (node_load + resource * prewarm_shortage) <= number_of_core_per_node:
            resource_time_arr[prewarm_location, :base_execution_time] += resource * prewarm_shortage
            node_load += resource * prewarm_shortage
            prewarm_shortage = 0
            disused_cold_startup = 0
        else:
            survived_cold_startup = (number_of_core_per_node - node_load) // resource
            disused_cold_startup = prewarm_shortage - survived_cold_startup
            resource_time_arr[prewarm_location, :base_execution_time] += resource * survived_cold_startup
            node_load += resource * survived_cold_startup
            cold_startup_rerouting_time = 0
            ##deadline check
            if delay_time + base_execution_time <= ddl_time:
                delay_time+=1 #the function can be executed within deadline
            else:
                delay_time=-1
                slo_violation_count = copy.deepcopy(disused_cold_startup)
                disused_cold_startup = 0
            prewarm_shortage = 0
    
    # print(number_of_invocation)
    return (resource_time_arr, slo_violation_count, disused_cold_startup, delay_time)

def calculate_resource_time_v4(nodes_load, lut_list, func_id, number_of_invocation, prewarm_location, delay_time): 
    nodes_load_list = list(nodes_load)   
    number_of_node= len(nodes_load_list)
    base_execution_time = lut_list[func_id] #baseline execution time
    ddl_time = base_execution_time * ddl_laxity
    resource = resource_per_request #core CPU
    
    resource_time_arr = np.zeros((number_of_node, 900))
    slo_violation_count = 0

    prewarm_shortage = number_of_invocation

    ##check the ddl before execution
    if delay_time + base_execution_time > ddl_time and prewarm_shortage:
        delay_time=-1
        slo_violation_count = number_of_invocation
        disused_cold_startup = 0       
        prewarm_shortage = 0

    if prewarm_shortage:
        # prewarm_location = nodes_load_list.index(min(nodes_load_list))
        if (nodes_load_list[prewarm_location] + resource * prewarm_shortage) <= number_of_core_per_node:
            resource_time_arr[prewarm_location, :base_execution_time] += resource * prewarm_shortage
            nodes_load_list[prewarm_location] += resource * prewarm_shortage
            prewarm_shortage = 0
            disused_cold_startup = 0
        else:
            cold_startup_location = copy.deepcopy(prewarm_location)
            survived_cold_startup = (number_of_core_per_node - nodes_load_list[cold_startup_location]) // resource
            disused_cold_startup = prewarm_shortage - survived_cold_startup
            resource_time_arr[cold_startup_location, :base_execution_time] += resource * survived_cold_startup
            nodes_load_list[cold_startup_location] += resource * survived_cold_startup
            cold_startup_rerouting_time = 0
            ##deadline check
            if delay_time + base_execution_time <= ddl_time:
                delay_time+=1 #the function can be executed within deadline
            else:
                delay_time=-1
                slo_violation_count = copy.deepcopy(disused_cold_startup)
                disused_cold_startup = 0
            prewarm_shortage = 0
    
    # print(number_of_invocation)
    return (resource_time_arr, slo_violation_count, disused_cold_startup, delay_time)

def violation_minimum_policy(lut_list, func_distribution_arr, leftover_resource_time_arr, vm_distribution_arr):
    global first_interval_flag
    global this_interval_avail_resource_time
    global next_interval_avail_resource_time
    global number_of_node
    total_number_of_func_type = func_distribution_arr.shape[0]
    new_vm_distribution_arr = np.zeros((total_number_of_func_type, number_of_node))
    number_of_func_type = 0
    arrival_func_type_list = []
    # cumulative_resource_time_arr = np.array([0]*2*900)
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*900))
    cumulative_resource_time_arr[:,:900] += leftover_resource_time_arr
    slo_violation_arr = np.zeros((total_number_of_func_type, 2))
    epoch_length = func_distribution_arr.shape[1] - 1

    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id,0]!=0:
            number_of_func_type+=1
            new_vm_distribution_arr[func_id,:]=1
            arrival_func_type_list.append(func_id)

    for func_id in range(total_number_of_func_type):
        if vm_distribution_arr[func_id,0] == 1:
            if new_vm_distribution_arr[func_id,0] != 1:
                cumulative_resource_time_arr[:,:300]+=2
        else:
            if new_vm_distribution_arr[func_id,0] == 1:
                cumulative_resource_time_arr[:,:900]+=2
    # cumulative_resource_time_arr[:,:]=number_of_func_type*2 #unit: core
    vm_distribution_arr = copy.deepcopy(new_vm_distribution_arr)
    load_list = []
    
    for second_time in range(epoch_length):
        for func_id in arrival_func_type_list: #bug here
            number_of_invocation = func_distribution_arr[func_id,second_time+1]
            if number_of_invocation:
                load_list=[]
                for node_id in range(number_of_node):
                    load_list.append(max(cumulative_resource_time_arr[node_id,second_time:second_time+15]))
                resource_time_arr, slo_violation_count = calculate_resource_time_v1(load_list, lut_list, func_id, number_of_invocation)
            
                cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                slo_violation_arr[func_id,0]+=slo_violation_count
                slo_violation_arr[func_id,1]+=number_of_invocation
    load_list = []
    for i in range(number_of_node):
        load_list.append(max(cumulative_resource_time_arr[i,:])/64)
    # print(load_list)
    # exit()
    # x=3
    # print(cumulative_resource_time_arr[20*x:20+20*x])
    # print(cumulative_resource_time_arr[55])
    # slo_violation_rate = np.sum(slo_violation_arr[:,0])/np.sum(slo_violation_arr[:,1])
    total_invocation =  np.sum(slo_violation_arr[:,1]) 
    total_violation = np.sum(slo_violation_arr[:,0])
    return cumulative_resource_time_arr[:,:900], total_violation, total_invocation, cumulative_resource_time_arr[:,900:], vm_distribution_arr

def binary_search(lut_list, func_distribution_arr, leftover_resource_time_arr, vm_distribution_arr, epoch_count):
    global first_interval_flag
    global this_interval_avail_resource_time
    global next_interval_avail_resource_time
    global number_of_node
    total_number_of_func_type = func_distribution_arr.shape[0]
    new_vm_distribution_arr = np.zeros((total_number_of_func_type, number_of_node))
    number_of_func_type = 0
    arrival_func_type_list = []
    # cumulative_resource_time_arr = np.array([0]*2*900)
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*900))
    cumulative_resource_time_arr[:,:900] += leftover_resource_time_arr
    slo_violation_arr = np.zeros((total_number_of_func_type, 2))
    epoch_length = func_distribution_arr.shape[1] - 1

    #record prewarm func id
    prewarm_func_list = []
    func_invocation_list = []
    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id,0]!=0:
            prewarm_func_list.append(func_id)
            func_invocation_list.append(func_distribution_arr[func_id,0])
    #         print(func_id, func_distribution_arr[func_id,0], lut_list[func_id])
    # print(sum(func_invocation_list))
    
    #record search func id
    prewarm_location_list = []
    search_func_list = []
    for func_id in prewarm_func_list:
        if np.any(vm_distribution_arr[func_id,:] != 0):
            the_node_id = int(np.where(vm_distribution_arr[func_id,:]==1)[0][0])
            prewarm_location_list.append(the_node_id)
        else:
            prewarm_location_list.append(0)
            search_func_list.append(func_id)

    ##search start:
    for search_func_id in search_func_list:
        candidate_prewarm_location_list = copy.deepcopy(prewarm_location_list)
        #store the best results
        min_score=999999
        best_node_id=0
        for search_node_id in range(number_of_node):

            #reset background
            index = prewarm_func_list.index(search_func_id)
            candidate_prewarm_location_list[index] = search_node_id
            candidate_cumulative_resource_time_arr = copy.deepcopy(cumulative_resource_time_arr)
            candidate_slo_violation_arr = copy.deepcopy(slo_violation_arr)
            last_epoch_func_list = []

            #startup vm
            for func_id in prewarm_func_list:
                if func_id not in search_func_list: ##vm exist from last epoch, no need to start up
                    the_node_id = int(np.where(vm_distribution_arr[func_id,:]==1)[0][0])
                    last_epoch_func_list.append(func_id)
                    candidate_cumulative_resource_time_arr[the_node_id,:900]+=4 #idle
                else: ##no vm from last epoch, need to start up
                    index = prewarm_func_list.index(func_id)
                    the_node_id = candidate_prewarm_location_list[index]
                    candidate_cumulative_resource_time_arr[the_node_id,:900]+=4 #idle
                    candidate_cumulative_resource_time_arr[the_node_id,:vm_startup_time]+=4 #startup
            
            #shutdown unnecessary vm based on arrival prediction
            for func_id in range(total_number_of_func_type):
                if func_id not in prewarm_func_list:
                    if np.any(vm_distribution_arr[func_id,:] != 0):
                        previous_prewarm_location_list = list(np.nonzero(vm_distribution_arr[func_id,:]))
                        for node in previous_prewarm_location_list:
                            candidate_cumulative_resource_time_arr[node,:vm_shutdown_time]+=2 #shutdown
            
            #calculate startup delay deponds on node location
            vm_startup_time_list = []
            for func_id in prewarm_func_list:
                index = prewarm_func_list.index(func_id)
                node_location = candidate_prewarm_location_list[index]
                if node_location < 2:
                    vm_startup_time_list.append(5)  #intra-rack delay
                else:
                    vm_startup_time_list.append(15)

            queued_func_id_list=[]
            queued_func_req_list=[]
            queued_func_delay_list=[]
            #calculation
            load_list = []
            for second_time in range(epoch_length):
                ##process queued func first
                updated_queued_func_id_list=[]
                updated_queued_func_req_list=[]
                updated_queued_func_delay_list=[]
                for i in range(len(queued_func_id_list)):
                    func_id=queued_func_id_list[i]
                    number_of_invocation=queued_func_req_list[i]
                    delay_time=queued_func_delay_list[i]
                    index = prewarm_func_list.index(func_id)
                    if second_time >= vm_startup_time_list[index]:
                        load_list=[]
                        for node_id in range(number_of_node):
                            load_list.append(max(candidate_cumulative_resource_time_arr[node_id,second_time:second_time+15]))
                        resource_time_arr, slo_violation_count, disused_cold_startup, delay_time = calculate_resource_time_v4(load_list, lut_list, func_id, number_of_invocation, candidate_prewarm_location_list[prewarm_func_list.index(func_id)], delay_time)
                        candidate_cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                        candidate_slo_violation_arr[func_id,0]+=slo_violation_count
                        candidate_slo_violation_arr[func_id,1]+=number_of_invocation
                        if disused_cold_startup:
                            if delay_time > 0:
                                updated_queued_func_id_list.append(func_id)
                                updated_queued_func_req_list.append(disused_cold_startup)
                                updated_queued_func_delay_list.append(delay_time)
                            else:
                                print("Warning!!!")
                    else:
                        ##vm not warm up yet
                        if func_id not in last_epoch_func_list:
                            updated_queued_func_id_list.append(func_id)
                            updated_queued_func_req_list.append(number_of_invocation)
                            updated_queued_func_delay_list.append(delay_time+1)
                        else:
                            load_list=[]
                            for node_id in range(number_of_node):
                                load_list.append(max(candidate_cumulative_resource_time_arr[node_id,second_time:second_time+15]))
                            resource_time_arr, slo_violation_count, disused_cold_startup, delay_time = calculate_resource_time_v4(load_list, lut_list, func_id, number_of_invocation, candidate_prewarm_location_list[prewarm_func_list.index(func_id)], delay_time)
                            candidate_cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                            candidate_slo_violation_arr[func_id,0]+=slo_violation_count
                            candidate_slo_violation_arr[func_id,1]+=number_of_invocation
                            if disused_cold_startup:
                                if delay_time > 0:
                                    updated_queued_func_id_list.append(func_id)
                                    updated_queued_func_req_list.append(disused_cold_startup)
                                    updated_queued_func_delay_list.append(delay_time)
                                else:
                                    print("Warning!!!")
                
                ## update queue list
                queued_func_id_list = copy.deepcopy(updated_queued_func_id_list)
                queued_func_req_list = copy.deepcopy(updated_queued_func_req_list)
                queued_func_delay_list = copy.deepcopy(updated_queued_func_delay_list)

                ##process just-come-in func
                for i in range(len(prewarm_func_list)):
                    func_id = prewarm_func_list[i]
                    number_of_invocation = func_distribution_arr[func_id,second_time+1]
                    # print('yes')
                    if number_of_invocation:
                        if second_time >= vm_startup_time_list[i]:
                            delay_time = 0
                        else: #survived vm from last round
                            if func_id not in last_epoch_func_list:
                                delay_time = 1
                                queued_func_id_list.append(func_id)
                                queued_func_req_list.append(number_of_invocation)
                                queued_func_delay_list.append(delay_time)
                            else:
                                delay_time = 0
                        if delay_time == 0:
                            load_list=[]
                            for node_id in range(number_of_node):
                                load_list.append(max(candidate_cumulative_resource_time_arr[node_id,second_time:second_time+15]))
                            resource_time_arr, slo_violation_count, disused_cold_startup, delay_time = calculate_resource_time_v4(load_list, lut_list, func_id, number_of_invocation, candidate_prewarm_location_list[i], delay_time)
                            candidate_cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                            candidate_slo_violation_arr[func_id,0]+=slo_violation_count
                            candidate_slo_violation_arr[func_id,1]+=number_of_invocation
                            if disused_cold_startup:
                                if delay_time > 0:
                                    queued_func_id_list.append(func_id)
                                    queued_func_req_list.append(disused_cold_startup)
                                    queued_func_delay_list.append(delay_time)
                                else:
                                    print("Warning!!!")


            ##wrap up results
            joule_in_15min = 0
            for node_id in range(number_of_node):
                for time_step in range(900):
                    joule_in_15min+=calculate_power(candidate_cumulative_resource_time_arr[node_id,time_step])
            kwh_in_15min = joule_in_15min * 0.0000002778
            hour = epoch_count//4
            hour = hour%24
            total_energy = kwh_in_15min * cop * cop_factor_list[hour]
            total_energy_cost = total_energy * price_list[hour]
            carbon = total_energy * carbon_density_list[hour]
            system_load = np.mean(candidate_cumulative_resource_time_arr[:,:900])/64
            final_score = total_energy_cost+np.sum(candidate_slo_violation_arr[:,0])+sum(queued_func_req_list)
            # final_score = np.sum(candidate_slo_violation_arr[:,0])
            # print(candidate_prewarm_location_list, np.sum(candidate_slo_violation_arr[:,0])+sum(queued_func_req_list)) ##debug binary
            if final_score < min_score:
                # print('here', search_func_id, search_node_id)
                min_score = copy.deepcopy(final_score)
                best_node_id = copy.deepcopy(search_node_id)
            # print('here', search_func_id, search_node_id, final_score, total_energy_cost, np.sum(candidate_slo_violation_arr[:,0]), candidate_slo_violation_arr[search_func_id,0], np.where(func_distribution_arr[search_func_id,1:]!=0))

        index = prewarm_func_list.index(search_func_id)
        prewarm_location_list[index]=best_node_id
    ##search end
    # print(prewarm_location_list, np.where(candidate_slo_violation_arr[:,0]!=0))
    # print(prewarm_location_list) ##debug binary
    # if epoch_count==1:
    #     exit()

    #startup vm
    for func_id in prewarm_func_list:
        if func_id not in search_func_list: ##vm exist from last epoch, no need to start up
            the_node_id = int(np.where(vm_distribution_arr[func_id,:]==1)[0][0])
            last_epoch_func_list.append(func_id)
            cumulative_resource_time_arr[the_node_id,:900]+=4 #idle
        else: ##no vm from last epoch, need to start up
            index = prewarm_func_list.index(func_id)
            the_node_id = prewarm_location_list[index]
            cumulative_resource_time_arr[the_node_id,:900]+=4 #idle
            cumulative_resource_time_arr[the_node_id,:vm_startup_time]+=4 #startup
                
    #shutdown unnecessary vm based on arrival prediction
    for func_id in range(total_number_of_func_type):
        if func_id not in prewarm_func_list:
            if np.any(vm_distribution_arr[func_id,:] != 0):
                previous_prewarm_location_list = list(np.nonzero(vm_distribution_arr[func_id,:]))
                for node in previous_prewarm_location_list:
                    cumulative_resource_time_arr[node,:vm_shutdown_time]+=2 #shutdown

    ##calculate startup latency
    vm_startup_time_list = []
    for func_id in prewarm_func_list:
        index = prewarm_func_list.index(func_id)
        node_location = prewarm_location_list[index]
        if node_location < 2:
            vm_startup_time_list.append(5)  #intra-rack delay
        else:
            vm_startup_time_list.append(15)

    vm_distribution_arr = copy.deepcopy(new_vm_distribution_arr)
    queued_func_id_list=[]
    queued_func_req_list=[]
    queued_func_delay_list=[]
    load_list = []
    # print(prewarm_location_list, prewarm_func_list)
    for second_time in range(epoch_length):
        ##process queued func first
        updated_queued_func_id_list=[]
        updated_queued_func_req_list=[]
        updated_queued_func_delay_list=[]
        for i in range(len(queued_func_id_list)):
            func_id=queued_func_id_list[i]
            number_of_invocation=queued_func_req_list[i]
            delay_time=queued_func_delay_list[i]
            index = prewarm_func_list.index(func_id)
            if second_time >= vm_startup_time_list[index]:
                load_list=[]
                for node_id in range(number_of_node):
                    load_list.append(max(cumulative_resource_time_arr[node_id,second_time:second_time+15]))
                resource_time_arr, slo_violation_count, disused_cold_startup, delay_time = calculate_resource_time_v4(load_list, lut_list, func_id, number_of_invocation, prewarm_location_list[prewarm_func_list.index(func_id)], delay_time)
                cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                slo_violation_arr[func_id,0]+=slo_violation_count
                slo_violation_arr[func_id,1]+=number_of_invocation
                if disused_cold_startup:
                    if delay_time > 0:
                        updated_queued_func_id_list.append(func_id)
                        updated_queued_func_req_list.append(disused_cold_startup)
                        updated_queued_func_delay_list.append(delay_time)
                    else:
                        print("Warning!!!")
                        exit()
            else:
                ##vm not warm up yet
                if func_id not in last_epoch_func_list:
                    updated_queued_func_id_list.append(func_id)
                    updated_queued_func_req_list.append(number_of_invocation)
                    updated_queued_func_delay_list.append(delay_time+1)
                else:
                    load_list=[]
                    for node_id in range(number_of_node):
                        load_list.append(max(cumulative_resource_time_arr[node_id,second_time:second_time+15]))
                    resource_time_arr, slo_violation_count, disused_cold_startup, delay_time = calculate_resource_time_v4(load_list, lut_list, func_id, number_of_invocation, prewarm_location_list[prewarm_func_list.index(func_id)], delay_time)
                    cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                    slo_violation_arr[func_id,0]+=slo_violation_count
                    slo_violation_arr[func_id,1]+=number_of_invocation
                    if disused_cold_startup:
                        if delay_time > 0:
                            updated_queued_func_id_list.append(func_id)
                            updated_queued_func_req_list.append(disused_cold_startup)
                            updated_queued_func_delay_list.append(delay_time)
                        else:
                            print("Warning!!!")
        
        ## update queue list
        queued_func_id_list = copy.deepcopy(updated_queued_func_id_list)
        queued_func_req_list = copy.deepcopy(updated_queued_func_req_list)
        queued_func_delay_list = copy.deepcopy(updated_queued_func_delay_list)

        ## process just-come-in func
        for i in range(len(prewarm_func_list)):
            func_id = prewarm_func_list[i]
            number_of_invocation = func_distribution_arr[func_id,second_time+1]
            # print('yes')
            if number_of_invocation:
                if second_time >= vm_startup_time_list[i]:
                    delay_time = 0
                else: #survived vm from last round
                    if func_id not in last_epoch_func_list:
                        delay_time = 1
                        queued_func_id_list.append(func_id)
                        queued_func_req_list.append(number_of_invocation)
                        queued_func_delay_list.append(delay_time)
                    else:
                        delay_time = 0
                if delay_time == 0:
                    load_list=[]
                    for node_id in range(number_of_node):
                        load_list.append(max(cumulative_resource_time_arr[node_id,second_time:second_time+15]))
                    resource_time_arr, slo_violation_count, disused_cold_startup, delay_time = calculate_resource_time_v4(load_list, lut_list, func_id, number_of_invocation, prewarm_location_list[i], delay_time)
                    cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                    slo_violation_arr[func_id,0]+=slo_violation_count
                    slo_violation_arr[func_id,1]+=number_of_invocation
                    if disused_cold_startup:
                        if delay_time > 0:
                            queued_func_id_list.append(func_id)
                            queued_func_req_list.append(disused_cold_startup)
                            queued_func_delay_list.append(delay_time)
                        else:
                            print("Warning!!!")

    load_list = []
    for i in range(number_of_node):
        load_list.append(max(cumulative_resource_time_arr[i,:])/64)
    # print(load_list)
    # exit()
    # x=3
    # print(cumulative_resource_time_arr[20*x:20+20*x])
    # print(cumulative_resource_time_arr[55])
    # slo_violation_rate = np.sum(slo_violation_arr[:,0])/np.sum(slo_violation_arr[:,1])
    total_invocation =  np.sum(slo_violation_arr[:,1]) 
    total_violation = np.sum(slo_violation_arr[:,0]) + sum(queued_func_req_list)
    print(np.sum(slo_violation_arr[:,0]), sum(queued_func_req_list))
    return cumulative_resource_time_arr[:,:900], total_violation, total_invocation, cumulative_resource_time_arr[:,900:], vm_distribution_arr

def score_policy(lut_list, func_distribution_arr, leftover_resource_time_arr, vm_distribution_arr):
    global first_interval_flag
    global this_interval_avail_resource_time
    global next_interval_avail_resource_time
    global number_of_node
    total_number_of_func_type = func_distribution_arr.shape[0]
    new_vm_distribution_arr = copy.deepcopy(vm_distribution_arr)
    new_vm_distribution_arr[:,:]=0
    number_of_func_type = 0
    arrival_func_type_list = []
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*epoch_length)) ##two times of epoch length means considering left over resource usage for next epoch
    cumulative_resource_time_arr[:,:epoch_length] += leftover_resource_time_arr
    slo_violation_arr = np.zeros((total_number_of_func_type, 2))

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

    #startup vm  based on cpu scoring
    prewarm_location_list = []
    last_epoch_func_list = []
    load_list=[0]*number_of_node
    index_count=0
    # for func_id in scored_prewarm_func_list:
    #     if np.any(vm_distribution_arr[func_id,:] != 0): ##vm exist from last epoch, no need to start up
    #         the_node_id = int(np.where(vm_distribution_arr[func_id,:]==1)[0][0])
    #         prewarm_location_list.append(the_node_id)
    #         last_epoch_func_list.append(func_id)
    #         new_vm_distribution_arr[func_id,the_node_id] = 1
    #         cumulative_resource_time_arr[the_node_id,:900]+=4 #idle
    #         load_list[the_node_id]+=scored_invocation_count_list[index_count]
    #     else: ##no vm from last epoch, need to start up
    #         the_node_id=load_list.index(min(load_list))
    #         prewarm_location_list.append(the_node_id)
    #         new_vm_distribution_arr[func_id,the_node_id] = 1
    #         cumulative_resource_time_arr[the_node_id,:900]+=4 #idle
    #         cumulative_resource_time_arr[the_node_id,:vm_startup_time]+=4 #startup
    #         load_list[the_node_id]+=scored_invocation_count_list[index_count]
    #     index_count+=1
        
    for func_id in scored_prewarm_func_list:
        the_node_id=index_count
        prewarm_location_list.append(the_node_id)
        new_vm_distribution_arr[func_id,the_node_id+1] = 1
        new_vm_distribution_arr[func_id,0]+=1
        cumulative_resource_time_arr[the_node_id,:epoch_length]+=idle_res_usage #idle
        # cumulative_resource_time_arr[the_node_id,:vm_startup_time]+=4 #startup
        index_count+=1
        index_count%=number_of_node

    #shutdown unnecessary vm based on arrival prediction
    for func_id in range(total_number_of_func_type):
        if func_id not in scored_prewarm_func_list:
            if np.any(vm_distribution_arr[func_id,0] != 0):
                previous_prewarm_location_list = list(np.nonzero(vm_distribution_arr[func_id,1:])[0])
                for node in previous_prewarm_location_list:
                    cumulative_resource_time_arr[node,:vm_shutdown_time]+=idle_res_usage #shutdown

    ##calculate startup latency
    vm_startup_time_list = []
    number_of_container_per_node = np.count_nonzero(new_vm_distribution_arr[:,1:], axis=0)
    # print(number_of_container_per_node)
    for node_id in range(number_of_node):
        number_of_coloation_container = number_of_container_per_node[node_id]
        if node_id < 2:
            vm_startup_time_list.append(2*number_of_coloation_container)  #intra-rack delay
        else:
            vm_startup_time_list.append(5*number_of_coloation_container)
    # print('number of containers:', number_of_container_per_node)
    # exit()

    vm_distribution_arr = copy.deepcopy(new_vm_distribution_arr)
    queued_func_id_list=[]
    queued_func_req_list=[]
    queued_func_delay_list=[]
    load_list = []
    # print(prewarm_location_list, prewarm_func_list)
    for second_time in range(epoch_length):
        ##process queued func first
        updated_queued_func_id_list=[]
        updated_queued_func_req_list=[]
        updated_queued_func_delay_list=[]
        for i in range(len(queued_func_id_list)):
            func_id=queued_func_id_list[i]
            number_of_invocation=queued_func_req_list[i]
            delay_time=queued_func_delay_list[i]
            # container_location = np.nonzero(vm_distribution_arr[func_id,1:])[0][0]
            container_location = prewarm_location_list[scored_prewarm_func_list.index(func_id)]
            if second_time >= vm_startup_time_list[container_location]:
                load_list=[]
                for node_id in range(number_of_node):
                    load_list.append(max(cumulative_resource_time_arr[node_id,second_time:second_time+15]))
                resource_time_arr, slo_violation_count, disused_cold_startup, delay_time = calculate_resource_time_v4(load_list, lut_list, func_id, number_of_invocation, prewarm_location_list[scored_prewarm_func_list.index(func_id)], delay_time)
                cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                slo_violation_arr[func_id,0]+=slo_violation_count
                slo_violation_arr[func_id,1]+=number_of_invocation
                if disused_cold_startup:
                    if delay_time > 0:
                        updated_queued_func_id_list.append(func_id)
                        updated_queued_func_req_list.append(disused_cold_startup)
                        updated_queued_func_delay_list.append(delay_time)
                    else:
                        print("Warning!!!")
                        exit()
            else:
                ##vm not warm up yet
                if func_id not in last_epoch_func_list:
                    updated_queued_func_id_list.append(func_id)
                    updated_queued_func_req_list.append(number_of_invocation)
                    updated_queued_func_delay_list.append(delay_time+1)
                else:
                    load_list=[]
                    for node_id in range(number_of_node):
                        load_list.append(max(cumulative_resource_time_arr[node_id,second_time:second_time+15]))
                    resource_time_arr, slo_violation_count, disused_cold_startup, delay_time = calculate_resource_time_v4(load_list, lut_list, func_id, number_of_invocation, prewarm_location_list[scored_prewarm_func_list.index(func_id)], delay_time)
                    cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                    slo_violation_arr[func_id,0]+=slo_violation_count
                    slo_violation_arr[func_id,1]+=number_of_invocation
                    if disused_cold_startup:
                        if delay_time > 0:
                            updated_queued_func_id_list.append(func_id)
                            updated_queued_func_req_list.append(disused_cold_startup)
                            updated_queued_func_delay_list.append(delay_time)
                        else:
                            print("Warning!!!")
        
        ## update queue list
        queued_func_id_list = copy.deepcopy(updated_queued_func_id_list)
        queued_func_req_list = copy.deepcopy(updated_queued_func_req_list)
        queued_func_delay_list = copy.deepcopy(updated_queued_func_delay_list)

        ## process just-come-in func
        for i in range(len(scored_prewarm_func_list)):
            func_id = scored_prewarm_func_list[i]
            number_of_invocation = func_distribution_arr[func_id,second_time+1]
            container_location = np.nonzero(vm_distribution_arr[func_id,:])[0][0]
            # print('yes')
            if number_of_invocation:
                if second_time >= vm_startup_time_list[container_location]:
                    delay_time = 0
                else: #survived vm from last round
                    if func_id not in last_epoch_func_list:
                        delay_time = 1
                        queued_func_id_list.append(func_id)
                        queued_func_req_list.append(number_of_invocation)
                        queued_func_delay_list.append(delay_time)
                    else:
                        delay_time = 0
                if delay_time == 0:
                    load_list=[]
                    for node_id in range(number_of_node):
                        load_list.append(max(cumulative_resource_time_arr[node_id,second_time:second_time+15]))
                    resource_time_arr, slo_violation_count, disused_cold_startup, delay_time = calculate_resource_time_v4(load_list, lut_list, func_id, number_of_invocation, prewarm_location_list[i], delay_time)
                    cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                    slo_violation_arr[func_id,0]+=slo_violation_count
                    slo_violation_arr[func_id,1]+=number_of_invocation
                    if disused_cold_startup:
                        if delay_time > 0:
                            queued_func_id_list.append(func_id)
                            queued_func_req_list.append(disused_cold_startup)
                            queued_func_delay_list.append(delay_time)
                        else:
                            print("Warning!!!")

    load_list = []
    for i in range(number_of_node):
        load_list.append(max(cumulative_resource_time_arr[i,:])/64)
    # print(load_list)
    # exit()
    # x=3
    # print(cumulative_resource_time_arr[20*x:20+20*x])
    # print(cumulative_resource_time_arr[55])
    # slo_violation_rate = np.sum(slo_violation_arr[:,0])/np.sum(slo_violation_arr[:,1])
    total_invocation =  np.sum(slo_violation_arr[:,1]) 
    total_violation = np.sum(slo_violation_arr[:,0]) + sum(queued_func_req_list)
    return cumulative_resource_time_arr[:,:900], total_violation, total_invocation, cumulative_resource_time_arr[:,900:], vm_distribution_arr

def multi_score_policy(lut_list, func_distribution_arr, leftover_resource_time_arr, vm_distribution_arr):
    total_number_of_func_type = func_distribution_arr.shape[0]
    new_vm_distribution_arr = copy.deepcopy(vm_distribution_arr)
    new_vm_distribution_arr[:,:]=0
    
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*epoch_length)) ##two times of epoch length means considering left over resource usage for next epoch
    
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
    
    index_count=0
    ## decides number of container and container location per IDs, preexisting containers are ignored
    for i in range(len(scored_prewarm_func_list)):
        func_id = scored_prewarm_func_list[i]
        peak_invocation = np.max(func_distribution_arr[func_id,1:])
        number_of_container = peak_invocation * idle_res_usage//number_of_core_per_node +1 ##stop here
        number_of_container = min(number_of_container, number_of_node)
        # number_of_container = 1 ##test here
        for j in range(number_of_container):
            the_node_id=index_count
            new_vm_distribution_arr[func_id,the_node_id+1] = 1
            new_vm_distribution_arr[func_id,0]+=1
            leftover_resource_time_arr[the_node_id,:epoch_length]+=idle_res_usage
            index_count+=1
            index_count%=number_of_node

    func_priority_list = scored_prewarm_func_list
    cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, leftover_resource_time_arr, func_priority_list, lut_list)

    return cumulative_resource_time_arr[:,:900], total_violation, total_invocation, cumulative_resource_time_arr[:,900:], vm_distribution_arr

def train_q_table(lut_list, func_distribution_arr, leftover_resource_time_arr, vm_distribution_arr):
    total_number_of_func_type = func_distribution_arr.shape[0]
    new_vm_distribution_arr = copy.deepcopy(vm_distribution_arr)
    new_vm_distribution_arr[:,:]=0
    
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*epoch_length)) ##two times of epoch length means considering left over resource usage for next epoch
    
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
    
    total_invocation = sum(scored_invocation_count_list)
    system_load_list = [0]*number_of_node
    for i in range(len(scored_prewarm_func_list)):
        func_id = scored_prewarm_func_list[i]
        req_num = scored_invocation_count_list[i]
        state_arr = np.zeros((1+number_of_node))
        state_arr[0] = int(10*req_num/total_invocation)
        for j in range(number_of_node):
            state_arr[1+j] = system_load_list[j]
    index_count=0
    ## decides number of container and container location per IDs, preexisting containers are ignored
    for i in range(len(scored_prewarm_func_list)):
        func_id = scored_prewarm_func_list[i]
        peak_invocation = np.max(func_distribution_arr[func_id,1:])
        number_of_container = peak_invocation * idle_res_usage//number_of_core_per_node +1 ##stop here
        number_of_container = min(number_of_container, number_of_node)
        # number_of_container = 1 ##test here
        for j in range(number_of_container):
            the_node_id=index_count
            new_vm_distribution_arr[func_id,the_node_id+1] = 1
            new_vm_distribution_arr[func_id,0]+=1
            leftover_resource_time_arr[the_node_id,:epoch_length]+=idle_res_usage
            index_count+=1
            index_count%=number_of_node

    func_priority_list = scored_prewarm_func_list
    cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, leftover_resource_time_arr, func_priority_list, lut_list)

    return cumulative_resource_time_arr[:,:900], total_violation, total_invocation, cumulative_resource_time_arr[:,900:], vm_distribution_arr

def hybrid_policy(lut_list, func_distribution_arr, leftover_resource_time_arr, vm_distribution_arr, regular_vm_list, regular_vm_location):
    global first_interval_flag
    global this_interval_avail_resource_time
    global next_interval_avail_resource_time
    global number_of_node
    
    total_number_of_func_type = func_distribution_arr.shape[0]
    new_vm_distribution_arr = np.zeros((total_number_of_func_type, number_of_node))

    per_request_resource = 0.5 #core
    # cumulative_resource_time_arr = np.array([0]*2*900)
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*900))
    cumulative_resource_time_arr[:,:900] += leftover_resource_time_arr
    slo_violation_arr = np.zeros((total_number_of_func_type, 2))
    epoch_length = func_distribution_arr.shape[1] - 1

    prewarm_func_list = []
    prewarm_location_list = []
    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id,0]!=0:
            prewarm_func_list.append(func_id)
    
    last_epoch_func_list = []
    node_id=0
    for func_id in range(total_number_of_func_type):
        if func_id in prewarm_func_list:
            if func_id not in regular_vm_list:
                if np.any(vm_distribution_arr[func_id,:] != 0):
                    pass #vm already exist, no need to startup another vm for this func type
                    last_epoch_func_list.append(func_id)
                    # print(vm_distribution_arr[func_id,:])
                    the_node_id = int(np.where(vm_distribution_arr[func_id,:]==1)[0][0])
                    prewarm_location_list.append(the_node_id)
                    new_vm_distribution_arr[func_id,the_node_id] = 1
                    cumulative_resource_time_arr[the_node_id,:900]+=4 #idle

                else:
                    pass #no vm exists, start up a new vm in consistent hashing
                    new_vm_distribution_arr[func_id,node_id] = 1
                    prewarm_location_list.append(node_id)
                    cumulative_resource_time_arr[node_id,:900]+=4 #idle
                    cumulative_resource_time_arr[node_id,:vm_startup_time]+=4 #startup
                    node_id+=1
                    node_id=node_id%number_of_node
            else:
                index = regular_vm_list.index(func_id)
                regular_vm_node_id = regular_vm_location[index]
                new_vm_distribution_arr[func_id,regular_vm_node_id] = 1
                prewarm_location_list.append(regular_vm_node_id)
                cumulative_resource_time_arr[regular_vm_node_id,:900]+=4 #idle
        else:
            if func_id not in regular_vm_list:
                if np.any(vm_distribution_arr[func_id,:] != 0):
                    previous_prewarm_location_list = list(np.nonzero(vm_distribution_arr[func_id,:]))
                    for node in previous_prewarm_location_list:
                        cumulative_resource_time_arr[node,:vm_shutdown_time]+=2 #shutdown
            else:
                index = regular_vm_list.index(func_id)
                regular_vm_node_id = regular_vm_location[index]
                new_vm_distribution_arr[func_id,regular_vm_node_id] = 1
                prewarm_location_list.append(regular_vm_node_id)
                cumulative_resource_time_arr[regular_vm_node_id,:900]+=4 #idle

    # cumulative_resource_time_arr[:,:]=len(prewarm_func_list)*2 #unit: core
    # prewarm_location_list = [0]*len(prewarm_func_list)

    ##calculate startup latency
    vm_startup_time_list = []
    for func_id in prewarm_func_list:
        if func_id not in regular_vm_list:
            index = prewarm_func_list.index(func_id)
            node_location = prewarm_location_list[index]
            number_of_coloation_container = prewarm_location_list.count(node_location)
            if node_location < 2:
                vm_startup_time_list.append(2*number_of_coloation_container)  #intra-rack delay
            else:
                vm_startup_time_list.append(5*number_of_coloation_container)
        else:
            vm_startup_time_list.append(0)
    
    vm_distribution_arr = copy.deepcopy(new_vm_distribution_arr)
    queued_func_id_list=[]
    queued_func_req_list=[]
    queued_func_delay_list=[]
    load_list = []
    # print(prewarm_location_list, prewarm_func_list)
    for second_time in range(epoch_length):
        ##process queued func first
        updated_queued_func_id_list=[]
        updated_queued_func_req_list=[]
        updated_queued_func_delay_list=[]
        for i in range(len(queued_func_id_list)):
            func_id=queued_func_id_list[i]
            number_of_invocation=queued_func_req_list[i]
            delay_time=queued_func_delay_list[i]
            index = prewarm_func_list.index(func_id)
            if second_time >= vm_startup_time_list[index]:
                load_list=[]
                for node_id in range(number_of_node):
                    load_list.append(max(cumulative_resource_time_arr[node_id,second_time:second_time+15]))
                resource_time_arr, slo_violation_count, disused_cold_startup, delay_time = calculate_resource_time_v4(load_list, lut_list, func_id, number_of_invocation, prewarm_location_list[prewarm_func_list.index(func_id)], delay_time)
                cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                slo_violation_arr[func_id,0]+=slo_violation_count
                slo_violation_arr[func_id,1]+=number_of_invocation
                if disused_cold_startup:
                    if delay_time > 0:
                        updated_queued_func_id_list.append(func_id)
                        updated_queued_func_req_list.append(disused_cold_startup)
                        updated_queued_func_delay_list.append(delay_time)
                    else:
                        print("Warning!!!")
                        exit()
            else:
                ##vm not warm up yet
                if func_id not in last_epoch_func_list:
                    updated_queued_func_id_list.append(func_id)
                    updated_queued_func_req_list.append(number_of_invocation)
                    updated_queued_func_delay_list.append(delay_time+1)
                else:
                    load_list=[]
                    for node_id in range(number_of_node):
                        load_list.append(max(cumulative_resource_time_arr[node_id,second_time:second_time+15]))
                    resource_time_arr, slo_violation_count, disused_cold_startup, delay_time = calculate_resource_time_v4(load_list, lut_list, func_id, number_of_invocation, prewarm_location_list[prewarm_func_list.index(func_id)], delay_time)
                    cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                    slo_violation_arr[func_id,0]+=slo_violation_count
                    slo_violation_arr[func_id,1]+=number_of_invocation
                    if disused_cold_startup:
                        if delay_time > 0:
                            updated_queued_func_id_list.append(func_id)
                            updated_queued_func_req_list.append(disused_cold_startup)
                            updated_queued_func_delay_list.append(delay_time)
                        else:
                            print("Warning!!!")
        
        ## update queue list
        queued_func_id_list = copy.deepcopy(updated_queued_func_id_list)
        queued_func_req_list = copy.deepcopy(updated_queued_func_req_list)
        queued_func_delay_list = copy.deepcopy(updated_queued_func_delay_list)

        ## process just-come-in func
        for i in range(len(prewarm_func_list)):
            func_id = prewarm_func_list[i]
            number_of_invocation = func_distribution_arr[func_id,second_time+1]
            # print('yes')
            if number_of_invocation:
                if second_time >= vm_startup_time_list[i]:
                    delay_time = 0
                else: #survived vm from last round
                    if func_id not in last_epoch_func_list:
                        delay_time = 1
                        queued_func_id_list.append(func_id)
                        queued_func_req_list.append(number_of_invocation)
                        queued_func_delay_list.append(delay_time)
                    else:
                        delay_time = 0
                if delay_time == 0:
                    load_list=[]
                    for node_id in range(number_of_node):
                        load_list.append(max(cumulative_resource_time_arr[node_id,second_time:second_time+15]))
                    resource_time_arr, slo_violation_count, disused_cold_startup, delay_time = calculate_resource_time_v4(load_list, lut_list, func_id, number_of_invocation, prewarm_location_list[i], delay_time)
                    cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                    slo_violation_arr[func_id,0]+=slo_violation_count
                    slo_violation_arr[func_id,1]+=number_of_invocation
                    if disused_cold_startup:
                        if delay_time > 0:
                            queued_func_id_list.append(func_id)
                            queued_func_req_list.append(disused_cold_startup)
                            queued_func_delay_list.append(delay_time)
                        else:
                            print("Warning!!!")

    load_list = []
    for i in range(number_of_node):
        load_list.append(max(cumulative_resource_time_arr[i,:])/64)
    # print(load_list)
    # exit()
    # x=3
    # print(cumulative_resource_time_arr[20*x:20+20*x])
    # print(cumulative_resource_time_arr[55])
    # slo_violation_rate = np.sum(slo_violation_arr[:,0])/np.sum(slo_violation_arr[:,1])
    total_invocation =  np.sum(slo_violation_arr[:,1])
    total_violation = np.sum(slo_violation_arr[:,0]) + sum(queued_func_req_list)
    return cumulative_resource_time_arr[:,:900], total_violation, total_invocation, cumulative_resource_time_arr[:,900:], vm_distribution_arr

def multi_hybrid_policy(lut_list, func_distribution_arr, leftover_resource_time_arr, vm_distribution_arr, regular_vm_list, regular_vm_location):
    total_number_of_func_type = func_distribution_arr.shape[0]
    new_vm_distribution_arr = copy.deepcopy(vm_distribution_arr)
    new_vm_distribution_arr[:,:]=0
    
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*epoch_length)) ##two times of epoch length means considering left over resource usage for next epoch
    
    #record prewarm func id
    prewarm_func_list = []
    func_invocation_list = []
    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id,0]!=0:
            prewarm_func_list.append(func_id)
            func_invocation_list.append(func_distribution_arr[func_id,0])
    
    index_count=0
    ## decides number of container and container location per IDs, preexisting containers are ignored
    node_id=0
    for func_id in range(total_number_of_func_type):
        if func_id in prewarm_func_list:
            if func_id not in regular_vm_list:
                peak_invocation = np.max(func_distribution_arr[func_id,1:])
                number_of_container = peak_invocation * idle_res_usage//number_of_core_per_node +1 ##stop here
                number_of_container = min(number_of_container, number_of_node)
                # number_of_container = 1 ##test here
                for j in range(number_of_container):
                    new_vm_distribution_arr[func_id,node_id+1] = 1
                    new_vm_distribution_arr[func_id,0]+=1
                    leftover_resource_time_arr[node_id,:epoch_length]+=idle_res_usage
                    node_id+=1
                    node_id%=number_of_node
        if func_id in regular_vm_list:
            index = regular_vm_list.index(func_id)
            regular_vm_node_id = regular_vm_location[index]
            new_vm_distribution_arr[func_id,regular_vm_node_id+1] = 1
            new_vm_distribution_arr[func_id,0] = 1
            leftover_resource_time_arr[regular_vm_node_id,:epoch_length]+=idle_res_usage

    #record prewarm func id based on scoring
    scored_prewarm_func_list = []
    scored_invocation_count_list = []
    while len(func_invocation_list):
        max_invocation = max(func_invocation_list)
        max_invocation_index = func_invocation_list.index(max_invocation)
        scored_invocation_count_list.append(func_invocation_list.pop(max_invocation_index))
        scored_prewarm_func_list.append(prewarm_func_list.pop(max_invocation_index))
        
    func_priority_list = copy.deepcopy(scored_prewarm_func_list)
    for func_id in regular_vm_list:
        if func_id not in func_priority_list:
            func_priority_list.insert(0, func_id)
    
    cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, leftover_resource_time_arr, func_priority_list, lut_list)

    return cumulative_resource_time_arr[:,:900], total_violation, total_invocation, cumulative_resource_time_arr[:,900:], vm_distribution_arr

def load_minimum_policy(lut_list, func_distribution_arr, leftover_resource_time_arr, vm_distribution_arr):
    global first_interval_flag
    global this_interval_avail_resource_time
    global next_interval_avail_resource_time
    global number_of_node
    total_number_of_func_type = func_distribution_arr.shape[0]
    new_vm_distribution_arr = np.zeros((total_number_of_func_type, number_of_node))
    per_request_resource = 0.5 #core
    # cumulative_resource_time_arr = np.array([0]*2*900)
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*900))
    cumulative_resource_time_arr[:,:900] += leftover_resource_time_arr
    slo_violation_arr = np.zeros((total_number_of_func_type, 2))
    epoch_length = func_distribution_arr.shape[1] - 1

    prewarm_func_list = []
    prewarm_location_list = []
    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id,0]!=0:
            prewarm_func_list.append(func_id)

    load_consolidation = 1
    peak_invocation_rate = max(func_distribution_arr[:,0])
    if peak_invocation_rate*per_request_resource + 2 > 64*number_of_node/len(prewarm_func_list):
        load_consolidation = 0
    
    if load_consolidation:
        node_id = 0
        for func_id in range(total_number_of_func_type):
            if func_id in prewarm_func_list:
                if np.any(vm_distribution_arr[func_id,:] != 0):
                    previous_prewarm_location_list = np.nonzero(vm_distribution_arr[func_id,:])[0]
                    # print('sirui', previous_prewarm_location_list, previous_prewarm_location_list[0])
                    prewarm_location_list.append(int(previous_prewarm_location_list[0]))
                    new_vm_distribution_arr[func_id,previous_prewarm_location_list[0]]=1
                    if len(previous_prewarm_location_list) > 1:
                        for node in previous_prewarm_location_list[1:]:
                            cumulative_resource_time_arr[node,:300]+=2
                else:
                    cumulative_resource_time_arr[node_id,:900]+=4
                    prewarm_location_list.append(node_id)
                    new_vm_distribution_arr[func_id,node_id]=1
            else:
                if np.any(vm_distribution_arr[func_id,:] != 0):
                    previous_prewarm_location_list = list(np.nonzero(vm_distribution_arr[func_id,:]))
                    for node in previous_prewarm_location_list:
                        cumulative_resource_time_arr[node,:300]+=2
            node_id+=1
            node_id=node_id%number_of_node
    else:
        for func_id in range(total_number_of_func_type):
            if func_id in prewarm_func_list:
                new_vm_distribution_arr[func_id,:]=1
                if np.any(vm_distribution_arr[func_id,:] != 0):
                    for node in range(number_of_node): 
                        if vm_distribution_arr[func_id, node] == 0:
                            cumulative_resource_time_arr[node,:900]+=4 #idle
                            cumulative_resource_time_arr[node,:300]+=4 #startup
                else:
                  cumulative_resource_time_arr[:,:900]+=4 #idle
                  cumulative_resource_time_arr[:,:300]+=4 #startup
            else:
                if np.any(vm_distribution_arr[func_id,:] != 0):
                    previous_prewarm_location_list = list(np.nonzero(vm_distribution_arr[func_id,:]))
                    for node in previous_prewarm_location_list:
                        cumulative_resource_time_arr[node,:300]+=2 #shutdown
        # cumulative_resource_time_arr[:,:]=len(prewarm_func_list)*2 #unit: core
        prewarm_location_list = [0]*len(prewarm_func_list)
    
    vm_distribution_arr = copy.deepcopy(new_vm_distribution_arr)
    for second_time in range(epoch_length):
        for i in range(len(prewarm_func_list)):
            func_id = prewarm_func_list[i]
            number_of_invocation = func_distribution_arr[func_id,second_time+1]
            if number_of_invocation:
                load_list=[]
                for node_id in range(number_of_node):
                    load_list.append(max(cumulative_resource_time_arr[node_id,second_time:second_time+15]))
                resource_time_arr, slo_violation_count = calculate_resource_time_v2(load_list, lut_list, func_id, number_of_invocation, load_consolidation, prewarm_location_list[i])
            
                cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                slo_violation_arr[func_id,0]+=slo_violation_count
                slo_violation_arr[func_id,1]+=number_of_invocation
    load_list = []
    for i in range(number_of_node):
        load_list.append(max(cumulative_resource_time_arr[i,:])/64)
    # print(load_list)
    # exit()
    # x=3
    # print(cumulative_resource_time_arr[20*x:20+20*x])
    # print(cumulative_resource_time_arr[55])
    # slo_violation_rate = np.sum(slo_violation_arr[:,0])/np.sum(slo_violation_arr[:,1])
    total_invocation =  np.sum(slo_violation_arr[:,1])
    total_violation = np.sum(slo_violation_arr[:,0])
    return cumulative_resource_time_arr[:,:900], total_violation, total_invocation, cumulative_resource_time_arr[:,900:], vm_distribution_arr

def ideal_policy(lut_list, func_distribution_arr, leftover_resource_time_arr):
    global first_interval_flag
    global this_interval_avail_resource_time
    global next_interval_avail_resource_time
    global number_of_node
    total_number_of_func_type = func_distribution_arr.shape[0]
    number_of_func_type = 0
    # cumulative_resource_time_arr = np.array([0]*2*900)
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*900))
    cumulative_resource_time_arr[:,:900] += leftover_resource_time_arr
    slo_violation_arr = np.zeros((total_number_of_func_type, 2))
    epoch_length = func_distribution_arr.shape[1] - 1

    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id,0]!=0:
            number_of_func_type+=1
    load_list = []
    
    for second_time in range(epoch_length):
        for func_id in range(number_of_func_type):
            number_of_invocation = func_distribution_arr[func_id,second_time+1]
            load_list=[]
            for node_id in range(number_of_node):
                load_list.append(max(cumulative_resource_time_arr[node_id,second_time:second_time+15]))
            resource_time_arr, slo_violation_count = calculate_resource_time_v1(load_list, lut_list, func_id, number_of_invocation)
        
            cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
            slo_violation_arr[func_id,0]+=slo_violation_count
            slo_violation_arr[func_id,1]+=number_of_invocation
    load_list = []
    for i in range(number_of_node):
        load_list.append(max(cumulative_resource_time_arr[i,:])/64)
    # print(load_list)
    # exit()
    # x=3
    # print(cumulative_resource_time_arr[20*x:20+20*x])
    # print(cumulative_resource_time_arr[55])
    # slo_violation_rate = np.sum(slo_violation_arr[:,0])/np.sum(slo_violation_arr[:,1])
    total_invocation =  np.sum(slo_violation_arr[:,1]) 
    total_violation = np.sum(slo_violation_arr[:,0])
    return cumulative_resource_time_arr[:,:900], total_violation, total_invocation, cumulative_resource_time_arr[:,900:]

def update_state(load_level_list, req_num, next_req_num, total_invocation, action_bits):
    found_first_container = 0
    side_req_num, left_req_num = divmod(req_num, sum(action_bits))
    main_req_num = side_req_num + left_req_num
    # print(main_req_num, side_req_num)
    # req_portion = (10*next_req_num)//total_invocation #ceiling to 10%, 20%, 30%, ....., 100%
    req_portion = int((next_req_num/total_invocation)//0.25)
    old_load_level_list = copy.deepcopy(load_level_list)
    next_state = req_portion

    for node_id in range(number_of_node):
        if action_bits[node_id]:
            if found_first_container:
                load_level_list[node_id]+=side_req_num
            else:
                found_first_container=1
                load_level_list[node_id]+=main_req_num
        load_level = load_level_list[node_id]
        abs_load_level = int((number_of_node*load_level/total_invocation)//0.25)
        if abs_load_level > 3:
            abs_load_level = 3
        # print(abs_load_level)
        next_state+=abs_load_level*pow(4,node_id+1)
        
    # print('update', req_num, old_load_level_list, action_bits, load_level_list)
    return next_state, load_level_list

def update_vm_distribution(new_vm_distribution_arr, func_id, action_bits):
    for node_id in range(number_of_node):
        if action_bits[node_id]:
            new_vm_distribution_arr[func_id,node_id+1] = 1
    new_vm_distribution_arr[func_id,0]=np.count_nonzero(new_vm_distribution_arr[func_id,1:])
    return new_vm_distribution_arr

def q_training(lut_list, func_distribution_arr, leftover_resource_time_arr, vm_distribution_arr, Q_table, learning_rate, discount_factor, exploration_prob, n_actions):
    total_number_of_func_type = func_distribution_arr.shape[0]
    new_vm_distribution_arr = copy.deepcopy(vm_distribution_arr)
    # new_vm_distribution_arr[:,:]=0
    search_leftover_resource_time_arr=copy.deepcopy(leftover_resource_time_arr)
    
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*epoch_length)) ##two times of epoch length means considering left over resource usage for next epoch
    
    #record prewarm func id
    prewarm_func_list = []
    func_invocation_list = []
    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id,0]!=0:
            prewarm_func_list.append(func_id)
            func_invocation_list.append(func_distribution_arr[func_id,0])
        else:
            new_vm_distribution_arr[func_id,:]=0
    
    #record prewarm func id based on scoring
    scored_prewarm_func_list = []
    scored_invocation_count_list = []
    while len(func_invocation_list):
        max_invocation = max(func_invocation_list)
        max_invocation_index = func_invocation_list.index(max_invocation)
        scored_invocation_count_list.append(func_invocation_list.pop(max_invocation_index))
        scored_prewarm_func_list.append(prewarm_func_list.pop(max_invocation_index))
        
    func_priority_list = scored_prewarm_func_list
    load_level_list = [0]* number_of_node
    total_invocation = int(sum(scored_invocation_count_list))
    # print(scored_invocation_count_list[0]/total_invocation)

    func_id = scored_prewarm_func_list[0]
    req_num = scored_invocation_count_list[0]
    next_req_num = scored_invocation_count_list[0+1]
    req_portion = int((req_num/total_invocation)//0.25) #ceiling to 0%, 10%, 20%, 30%, ....., 90%
    current_state = req_portion
    reward = 0
    # print('initial', current_state, req_num, total_invocation)
    for node_id in range(number_of_node):
        load_level = load_level_list[node_id]
        abs_load_level = int((number_of_node*load_level/total_invocation)//0.25)
        if abs_load_level > 3:
            abs_load_level = 3
        current_state+=abs_load_level*pow(4,node_id+1)

    for i in range(0, len(scored_prewarm_func_list)+1):
        
        # Choose action with epsilon-greedy strategy
        if np.random.rand() < exploration_prob:
            action = np.random.randint(0, n_actions) # Explore
        else:
            action = np.argmax(Q_table[current_state]) # Exploit
        
        # Simulate the environment (move to the next state)
        # For simplicity, move to the next state
        action_bits = np.array(list(np.binary_repr(action+1, width=number_of_node)), dtype=int)
        next_state, load_level_list = update_state(load_level_list, req_num, next_req_num, total_invocation, action_bits)
        new_vm_distribution_arr = update_vm_distribution(new_vm_distribution_arr, func_id, action_bits)

        # Define a simple reward function (1 if the goal state is reached, 0 otherwise)
        search_leftover_resource_time_arr = update_leftover(new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
        search_cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
            vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
        
        # reward = 1 - total_violation/total_invocation
        # reward = 1 - slo_violation_arr[func_id,0]/total_invocation
        reward += (slo_violation_arr[func_id,1] - slo_violation_arr[func_id,0])/total_invocation
        # print(req_num/total_invocation, action_bits, reward)

        # Update Q-value using the Q-learning update rule
        Q_table[current_state, action] += learning_rate * \
            (reward + discount_factor *
             np.max(Q_table[next_state]) - Q_table[current_state, action])
        
        # print(np.max(Q_table),np.min(Q_table))

        current_state = next_state  # Move to the next state
        if i < len(scored_prewarm_func_list):
            func_id = scored_prewarm_func_list[i]
            req_num = scored_invocation_count_list[i]
            next_req_num = scored_invocation_count_list[i]
        elif i == len(scored_prewarm_func_list)+1:
            func_id = scored_prewarm_func_list[i]
            req_num = scored_invocation_count_list[i]
            next_req_num = 0
    
    # print('end search')
    leftover_resource_time_arr = update_leftover(new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
    cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, leftover_resource_time_arr, func_priority_list, lut_list)

    return cumulative_resource_time_arr[:,:900], total_violation, total_invocation, cumulative_resource_time_arr[:,900:], new_vm_distribution_arr

def q_testing(lut_list, func_distribution_arr, leftover_resource_time_arr, vm_distribution_arr, Q_table, learning_rate, discount_factor, exploration_prob, n_actions):
    total_number_of_func_type = func_distribution_arr.shape[0]
    new_vm_distribution_arr = copy.deepcopy(vm_distribution_arr)
    # new_vm_distribution_arr[:,:]=0
    search_leftover_resource_time_arr=copy.deepcopy(leftover_resource_time_arr)
    
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*epoch_length)) ##two times of epoch length means considering left over resource usage for next epoch
    
    #record prewarm func id
    prewarm_func_list = []
    func_invocation_list = []
    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id,0]!=0:
            prewarm_func_list.append(func_id)
            func_invocation_list.append(func_distribution_arr[func_id,0])
        else:
            new_vm_distribution_arr[func_id,:]=0
    
    #record prewarm func id based on scoring
    scored_prewarm_func_list = []
    scored_invocation_count_list = []
    while len(func_invocation_list):
        max_invocation = max(func_invocation_list)
        max_invocation_index = func_invocation_list.index(max_invocation)
        scored_invocation_count_list.append(func_invocation_list.pop(max_invocation_index))
        scored_prewarm_func_list.append(prewarm_func_list.pop(max_invocation_index))
        
    func_priority_list = scored_prewarm_func_list
    load_level_list = [0]* number_of_node
    total_invocation = int(sum(scored_invocation_count_list))

    func_id = scored_prewarm_func_list[0]
    req_num = scored_invocation_count_list[0]
    next_req_num = scored_invocation_count_list[0+1]
    req_portion = int((req_num/total_invocation)//0.25) #ceiling to 0%, 10%, 20%, 30%, ....., 90%
    current_state = req_portion
    # print('initial', current_state, req_num, total_invocation)
    for node_id in range(number_of_node):
        load_level = load_level_list[node_id]
        abs_load_level = int((number_of_node*load_level/total_invocation)//0.25)
        if abs_load_level > 3:
            abs_load_level = 3
        current_state+=abs_load_level*pow(5,node_id+1)

    for i in range(0, len(scored_prewarm_func_list)+1):
        
        action = np.random.randint(0, n_actions) # Explore

        
        # Simulate the environment (move to the next state)
        # For simplicity, move to the next state
        action_bits = np.array(list(np.binary_repr(action+1, width=number_of_node)), dtype=int)
        next_state, load_level_list = update_state(load_level_list, req_num, next_req_num, total_invocation, action_bits)
        new_vm_distribution_arr = update_vm_distribution(new_vm_distribution_arr, func_id, action_bits)

        current_state = next_state  # Move to the next state
        if i < len(scored_prewarm_func_list):
            func_id = scored_prewarm_func_list[i]
            req_num = scored_invocation_count_list[i]
            next_req_num = scored_invocation_count_list[i]
        elif i == len(scored_prewarm_func_list)+1:
            func_id = scored_prewarm_func_list[i]
            req_num = scored_invocation_count_list[i]
            next_req_num = 0
    
    # print('end search')
    leftover_resource_time_arr = update_leftover(new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
    cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, leftover_resource_time_arr, func_priority_list, lut_list)

    return cumulative_resource_time_arr[:,:900], total_violation, total_invocation, cumulative_resource_time_arr[:,900:], new_vm_distribution_arr

def slo_optimizer(lut_list, func_distribution_arr, leftover_resource_time_arr, vm_distribution_arr, decision_time):
    total_number_of_func_type = func_distribution_arr.shape[0]
    new_vm_distribution_arr = copy.deepcopy(vm_distribution_arr)
    # new_vm_distribution_arr[:,:]=0
    search_leftover_resource_time_arr=copy.deepcopy(leftover_resource_time_arr)
    
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*epoch_length)) ##two times of epoch length means considering left over resource usage for next epoch
    
    #record prewarm func id
    prewarm_func_list = []
    func_invocation_list = []
    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id,0]!=0:
            prewarm_func_list.append(func_id)
            func_invocation_list.append(func_distribution_arr[func_id,0])
        else:
            new_vm_distribution_arr[func_id,:]=0
    
    #record prewarm func id based on scoring
    scored_prewarm_func_list = []
    scored_invocation_count_list = []
    while len(func_invocation_list):
        max_invocation = max(func_invocation_list)
        max_invocation_index = func_invocation_list.index(max_invocation)
        scored_invocation_count_list.append(func_invocation_list.pop(max_invocation_index))
        scored_prewarm_func_list.append(prewarm_func_list.pop(max_invocation_index))
    
    # index_count=0
    # ## decides number of container and container location per IDs, preexisting containers are ignored
    # for i in range(len(scored_prewarm_func_list)):
    #     func_id = scored_prewarm_func_list[i]
    #     peak_invocation = np.max(func_distribution_arr[func_id,1:])
    #     number_of_container = peak_invocation * idle_res_usage//number_of_core_per_node +1 ##stop here
    #     number_of_container = min(number_of_container, number_of_node)
    #     # number_of_container = 1 ##test here
    #     for j in range(number_of_container):
    #         the_node_id=index_count
    #         new_vm_distribution_arr[func_id,the_node_id+1] = 1
    #         # new_vm_distribution_arr[func_id,0]+=1
    #         new_vm_distribution_arr[func_id,0]=np.count_nonzero(new_vm_distribution_arr[func_id,1:])   
    #         index_count+=1
    #         index_count%=number_of_node
    
        
    func_priority_list = scored_prewarm_func_list
    search_leftover_resource_time_arr = update_leftover(new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
    cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
    
    
    black_list = []
    search_history=[0]*total_number_of_func_type
    start_time = time.time()
    total_violation_count = copy.deepcopy(np.sum(slo_violation_arr[:,0]))
    while (time.time() - start_time) < decision_time:
        search_func_id = -1
        violation_count = -1
        for func_id in scored_prewarm_func_list:
            if func_id not in black_list:
                if slo_violation_arr[func_id,0] > violation_count:
                    search_func_id = copy.deepcopy(func_id)
                    violation_count = copy.deepcopy(slo_violation_arr[func_id,0])
        
        
        if search_func_id > -1:
        
            search_vm_distribution_arr = slo_optimization(new_vm_distribution_arr, search_func_id)
            
            search_leftover_resource_time_arr = update_leftover(search_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
            
            cumulative_resource_time_arr, total_violation, total_invocation, search_vm_distribution_arr, slo_violation_arr = simulation(
                vm_distribution_arr, search_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
            if np.sum(slo_violation_arr[:,0]) < total_violation_count:
                # print("search success", search_func_id, np.sum(slo_violation_arr[:,0]), total_violation_count)
                total_violation_count = copy.deepcopy(np.sum(slo_violation_arr[:,0]))
                # search_history[search_func_id]=0
                new_vm_distribution_arr=copy.deepcopy(search_vm_distribution_arr)
            else:
                # print("search fail", search_func_id, func_distribution_arr[search_func_id, 0], new_vm_distribution_arr[search_func_id,:], violation_count, search_vm_distribution_arr[search_func_id,:], slo_violation_arr[search_func_id, 0])
                search_history[search_func_id]+=1
                if search_history[search_func_id] > 15:
                    # print(search_func_id, "in black list")
                    black_list.append(search_func_id)
        else:
            print('all IDs in black list')
            break
        # print(i)
    
    # print('end search')
    leftover_resource_time_arr = update_leftover(new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
    cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, leftover_resource_time_arr, func_priority_list, lut_list)

    return cumulative_resource_time_arr[:,:900], total_violation, total_invocation, cumulative_resource_time_arr[:,900:], new_vm_distribution_arr

def dual_optimizer(lut_list, func_distribution_arr, leftover_resource_time_arr, vm_distribution_arr, epoch, decision_time, slo_constraint):
    total_number_of_func_type = func_distribution_arr.shape[0]
    new_vm_distribution_arr = copy.deepcopy(vm_distribution_arr)
    # new_vm_distribution_arr[:,:]=0
    search_leftover_resource_time_arr=copy.deepcopy(leftover_resource_time_arr)
    
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*epoch_length)) ##two times of epoch length means considering left over resource usage for next epoch
    
    #record prewarm func id
    prewarm_func_list = []
    func_invocation_list = []
    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id,0]!=0:
            prewarm_func_list.append(func_id)
            func_invocation_list.append(func_distribution_arr[func_id,0])
        else:
            new_vm_distribution_arr[func_id,:]=0
    
    #record prewarm func id based on scoring
    scored_prewarm_func_list = []
    scored_invocation_count_list = []
    while len(func_invocation_list):
        max_invocation = max(func_invocation_list)
        max_invocation_index = func_invocation_list.index(max_invocation)
        scored_invocation_count_list.append(func_invocation_list.pop(max_invocation_index))
        scored_prewarm_func_list.append(prewarm_func_list.pop(max_invocation_index))
        
    func_priority_list = scored_prewarm_func_list
    search_leftover_resource_time_arr = update_leftover(new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
    search_cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
    
    
    black_list = [-1]
    search_history=[0]*total_number_of_func_type
    start_time = time.time()
    search_energy_cost, search_carbon = calculate_power_cost(search_cumulative_resource_time_arr, epoch)
    total_violation_count = copy.deepcopy(np.sum(slo_violation_arr[:,0]))
    total_carbon_emission = copy.deepcopy(search_carbon)
    while (time.time() - start_time) < decision_time:
        search_func_id = -1
        if (total_violation/total_invocation) > slo_constraint:
            search_func_id = random.choice(scored_prewarm_func_list)
            if search_func_id > -1:
                search_vm_distribution_arr = slo_optimization(new_vm_distribution_arr, search_func_id)
                search_leftover_resource_time_arr = update_leftover(search_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
                
                search_cumulative_resource_time_arr, total_violation, total_invocation, search_vm_distribution_arr, slo_violation_arr = simulation(
                    vm_distribution_arr, search_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
                if np.sum(slo_violation_arr[:,0]) < total_violation_count:
                    # print("SLO success", search_func_id, np.sum(slo_violation_arr[:,0])/total_invocation, total_violation_count/total_invocation)
                    total_violation_count = copy.deepcopy(np.sum(slo_violation_arr[:,0]))
                    new_vm_distribution_arr=copy.deepcopy(search_vm_distribution_arr)
                    search_energy_cost, search_carbon = calculate_power_cost(search_cumulative_resource_time_arr, epoch)
                    total_carbon_emission = copy.deepcopy(search_carbon)
        else:
            search_func_id = random.choice(scored_prewarm_func_list)
            if search_func_id > -1:
                search_vm_distribution_arr = carbon_optimization(new_vm_distribution_arr, search_func_id)
                search_leftover_resource_time_arr = update_leftover(search_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
                
                search_cumulative_resource_time_arr, search_total_violation, search_total_invocation, search_vm_distribution_arr, slo_violation_arr = simulation(
                    vm_distribution_arr, search_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
                search_energy_cost, search_carbon = calculate_power_cost(search_cumulative_resource_time_arr, epoch)
                if search_carbon < total_carbon_emission and (search_total_violation/search_total_invocation) <= 0.05:
                    # print('Carbon success', search_func_id, search_carbon, total_carbon_emission)
                    total_carbon_emission = copy.deepcopy(search_carbon)
                    total_violation_count = copy.deepcopy(np.sum(slo_violation_arr[:,0]))
                    new_vm_distribution_arr=copy.deepcopy(search_vm_distribution_arr)   
    
    # print('end search')
    leftover_resource_time_arr = update_leftover(new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
    cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, leftover_resource_time_arr, func_priority_list, lut_list)

    return cumulative_resource_time_arr[:,:900], total_violation, total_invocation, cumulative_resource_time_arr[:,900:], new_vm_distribution_arr

def action_to_distribution(action_arr, scored_prewarm_func_list, new_vm_distribution_arr):
    number_of_active_func_id = len(scored_prewarm_func_list)
    for i in range(number_of_active_func_id):
        func_id = scored_prewarm_func_list[i]
        action = action_arr[i]
        action_bits = np.array(list(np.binary_repr(action, width=number_of_node)), dtype=int)
        new_vm_distribution_arr[func_id, 0] = sum(action_bits)
        new_vm_distribution_arr[func_id, 1:] = action_bits
        # print(func_id, action, action_bits, sum(action_bits))
    
    return new_vm_distribution_arr

def update_objective(search_cumulative_resource_time_arr, total_violation, total_invocation, epoch):
    objective_arr = np.zeros(3)
    objective_arr[0] = total_violation/total_invocation * 100

    joule_per_node = []
    for node_id in range(number_of_node):
        joule_in_15min = 0
        for time_step in range(900):
            joule_in_15min+=calculate_power(search_cumulative_resource_time_arr[node_id,time_step])
        joule_per_node.append(joule_in_15min)

    kwh_in_15min = sum(joule_per_node) * 0.0000002778
    energy_per_node = []
    for node_id in range(number_of_node):
        energy_per_node.append(joule_per_node[node_id] * 0.0000002778)
    hour = epoch//4
    hour = hour%24
    total_energy = kwh_in_15min * cop * cop_factor_list[hour]
    # print('J', joule_in_15min)
    # total_energy_cost = total_energy * price_list[hour]

    water_latent_heat = 0.66 #KWh/L
    water_latent_heat_per_KJ = 0.66 * 3600 / 0.0000002778
    cycle_of_concentration = 5

    carbon = total_energy * carbon_density_list[hour]

    system_load = np.mean(search_cumulative_resource_time_arr[:,:],axis=1)/number_of_core_per_node
    # print(system_load)
    per_node_cooling_efficiency = []

    # for node_id in range(number_of_node):
    #     cooling_level = round(system_load[node_id]/0.1)
    #     # per_node_cooling_efficiency.append(1 - cooling_level*0.125)
    #     if cooling_level > 4:
    #         cooling_level = 4
    #     per_node_cooling_efficiency.append(1 - cooling_level*0.2)

    # for node_id in range(number_of_node):
    #     if system_load[node_id] > 0.5:
    #         per_node_cooling_efficiency.append(0.3)
    #     elif system_load[node_id] > 0.3:
    #         per_node_cooling_efficiency.append(0.6)
    #     else:
    #         per_node_cooling_efficiency.append(1.2)
    
    for node_id in range(number_of_node):
        if energy_per_node[node_id] > 0.09:
            per_node_cooling_efficiency.append(0.3)
        elif energy_per_node[node_id] > 0.06:
            per_node_cooling_efficiency.append(0.6)
        else:
            per_node_cooling_efficiency.append(0.9)

    # print('cooling', energy_per_node, per_node_cooling_efficiency, np.array(energy_per_node)/np.array(per_node_cooling_efficiency))
    # exit()
    electricity_water = total_energy * water_density
    evaporation_water = 0
    for node_id in range(number_of_node):
        evaporation_water+=(energy_per_node[node_id]/per_node_cooling_efficiency[node_id]/water_latent_heat)
        # evaporation_water+=(joule_per_node[node_id]/water_latent_heat_per_KJ)
    # evaporation_water = total_energy / water_latent_heat #1KWh = 3600 KJ

    if total_energy > 0.4:
        cooling_efficiency = 0.9 #0.44
    elif total_energy > 0.3:
        cooling_efficiency = 0.6 #0.2
    else:
        cooling_efficiency = 0.9 #0.22
    # print('energy', total_energy)
    evaporation_water = total_energy / water_latent_heat / cooling_efficiency

    blowdown_water = evaporation_water / (cycle_of_concentration - 1)
    # print(electricity_water, evaporation_water+blowdown_water)

    objective_arr[1] = carbon
    objective_arr[2] = (electricity_water + evaporation_water + blowdown_water) * 100
    # system_load = np.mean(resource_usage_arr[:,:])/number_of_core_per_node
    return objective_arr

def update_population(child_designs, child_values, population, population_value, weight_arr, update_ratio_table):
    pass
    population_size = len(population)
    update_count = 0
    update_list = list(range(population_size))
    random.shuffle(update_list)
    length_of_update_list = len(update_list)

    child_list = list(range(len(child_designs)))
    random.shuffle(child_list)

    while child_list:
        index = child_list.pop()
        child_design = child_designs[index]
        child_value = child_values[index]
        j = 0
        while j < length_of_update_list:
        # for j in update_list:
            the_index = update_list[j]
            candidate = population_value[the_index]
            weight = weight_arr[the_index]
            replace = False

            fit_new = local_fit(child_value, weight) #the most promising optimization direction
            fit_old = local_fit(candidate, weight) #the most promising optimization direction
            if fit_new < fit_old:
                # print('update solution:', fit_new, fit_old, child_design_value, candidate)
                # print('update')
                replace = True
            if replace:
                population[the_index] = copy.deepcopy(child_design)
                population_value[the_index] = copy.deepcopy(child_value)
                update_ratio_table[the_index] = [1]
                update_count += 1
                updated = True
                j = length_of_update_list
            else:
                j+=1

    return population, population_value, update_ratio_table

def smart_brain_decision(weight_group, update_ratio_table):
    chosen_starting_points = []
    prediction = []
    for i in range(len(weight_group)):
        current_pareto_front = []
        for index in weight_group[i]:
            update_ratio_list = update_ratio_table[index]
            current_pareto_front.append(sum(update_ratio_list)/len(update_ratio_list))
            
        sorted_pre = sorted(range(len(current_pareto_front)), key=lambda i: current_pareto_front[i], reverse = True)
        index = sorted_pre[0]
        prediction.append(current_pareto_front[index])
        chosen_starting_points.append(weight_group[i][index])
    return chosen_starting_points, prediction

def crossover(parent_design_1, parent_design_2):
    child_1 = copy.deepcopy(parent_design_1)
    child_2 = copy.deepcopy(parent_design_2)
    number_of_rows = child_1.shape[0]
    # number_of_columns = self.number_of_city
    avail_row = list(range(number_of_rows))
    number_of_rows_to_be_exchanged = random.choices(list(range(1,number_of_rows)), k =1)[0]
    #exchange_list = random.sample(list(range(number_of_columns)), k = number_of_columns_to_be_exchanged)
    exchange_list = random.sample(avail_row, k = number_of_rows_to_be_exchanged)
    # print('cross over index:', exchange_list)
    
    # crossover
    for i in range(number_of_rows):
        if i in exchange_list: 
            child_1[i,:] = copy.deepcopy(parent_design_2[i,:])
            child_2[i,:] = copy.deepcopy(parent_design_1[i,:])
        else:
            child_1[i,:] = copy.deepcopy(parent_design_1[i,:])
            child_2[i,:] = copy.deepcopy(parent_design_2[i,:])
    return (child_1,child_2)

def GA(parent_design_1, parent_design_2):
    child_design_1, child_design_2 = crossover(parent_design_1, parent_design_2)
    # child_design_3 = perturb(step_size, child_design_1)
    # child_design_4 = perturb(step_size, child_design_2)
    return [child_design_1, child_design_2]

def hybrid_search(lut_list, func_distribution_arr, leftover_resource_time_arr, vm_distribution_arr, epoch, decision_time, slo_constraint, weight_arr):
    total_number_of_func_type = func_distribution_arr.shape[0]
    search_new_vm_distribution_arr = copy.deepcopy(vm_distribution_arr)
    search_new_vm_distribution_arr[:,:]=0
    search_leftover_resource_time_arr = copy.deepcopy(leftover_resource_time_arr)
    
    search_cumulative_resource_time_arr = np.zeros((number_of_node, 2*epoch_length)) ##two times of epoch length means considering left over resource usage for next epoch
    
    #initialize population 
    number_of_objective = 3
    standard_objective = [100]*number_of_objective
    population_size = 20
    population_value = []
    population = []
    update_ratio_table = []
    for i in range(population_size):
        update_ratio_table.append([1])

    
    #initialize weight group
    kmeans = KMeans(n_clusters=number_of_objective, random_state=0)
    cluster_list = kmeans.fit_predict(weight_arr)
    weight_group = []
    for i in range(number_of_objective):
        weight_group.append(list(np.where(cluster_list == i)[0]))

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
    
    func_priority_list = scored_prewarm_func_list
    number_active_func_id = len(func_priority_list)
    # print(number_active_func_id)

    action_arr = np.random.randint(0, pow(2,number_of_node), size=len(func_priority_list), dtype=int)
    observation_arr = np.zeros((len(func_priority_list),3)) #SLO rate, Req number, Runtime per function ID
    objective_arr = np.zeros(3)

    # # new_vm_distribution_arr = action_to_distribution(action_arr, scored_prewarm_func_list, new_vm_distribution_arr)
    # for i in range(number_active_func_id):
    #     func_id = scored_prewarm_func_list[i]
    #     observation_arr[i,1] = scored_invocation_count_list[i]
    #     observation_arr[i,2] = lut_list[func_id]

    index_count=0
    ## decides number of container and container location per IDs, preexisting containers are ignored
    for i in range(len(scored_prewarm_func_list)):
        func_id = scored_prewarm_func_list[i]
        peak_invocation = np.max(func_distribution_arr[func_id,1:])
        number_of_container = peak_invocation * idle_res_usage//number_of_core_per_node +1 ##stop here
        number_of_container = min(number_of_container, number_of_node)
        # number_of_container = 1 ##test here
        for j in range(number_of_container):
            the_node_id=index_count
            search_new_vm_distribution_arr[func_id,the_node_id+1] = 1
            search_new_vm_distribution_arr[func_id,0]+=1
            # search_leftover_resource_time_arr[the_node_id,:epoch_length]+=idle_res_usage
            index_count+=1
            index_count%=number_of_node

    # search_new_vm_distribution_arr[:,:]=0
    search_new_vm_distribution_arr = np.zeros((total_number_of_func_type,number_of_node+1),dtype=int)
    for i in range(population_size):
        random_action = np.random.randint(low=0, high=2, size=(number_active_func_id,number_of_node), dtype=int)
        search_new_vm_distribution_arr[scored_prewarm_func_list,1:] = random_action
        search_new_vm_distribution_arr[:,0] = np.sum(search_new_vm_distribution_arr[:,1:], axis=1)

        search_leftover_resource_time_arr = update_leftover(search_new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
        search_cumulative_resource_time_arr, total_violation, total_invocation, next_vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, search_new_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
        objective_arr = update_objective(search_cumulative_resource_time_arr, total_violation, total_invocation, epoch)
        
        population.append(copy.deepcopy(search_new_vm_distribution_arr))
        population_value.append(copy.deepcopy(objective_arr))

    for i in range(50):
        if i < 20:
            member_list = random.sample(list(range(population_size)), k = number_of_objective)
        else:
            for member_id in range(population_size):
                if len(update_ratio_table[member_id]) > 10:
                    update_ratio_table[member_id] = update_ratio_table[member_id][-10:]
            if random.random() < 0.9:
                member_list, prediction = smart_brain_decision(weight_group, update_ratio_table)
            else:
                member_list = random.sample(list(range(population_size)), k = number_of_objective)
        child_designs = []
        child_values = []
        # print(i, member_list)
        for member in member_list:
            search_time = 10
            update_time = 0
            for j in range(search_time):
                search_func_id = random.choice(scored_prewarm_func_list)
                search_new_vm_distribution_arr = local_search(population[member], search_func_id)

                search_leftover_resource_time_arr = update_leftover(search_new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
                search_cumulative_resource_time_arr, total_violation, total_invocation, next_vm_distribution_arr, slo_violation_arr = simulation(
                    vm_distribution_arr, search_new_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
                search_objective_arr = update_objective(search_cumulative_resource_time_arr, total_violation, total_invocation, epoch)

                if local_fit(search_objective_arr, weight_arr[member]) < local_fit(population_value[member], weight_arr[member]):
                    update_time += 1
                    population[member] = copy.deepcopy(search_new_vm_distribution_arr)
                    population_value[member] = copy.deepcopy(search_objective_arr)

            update_ratio_table[member].append(update_time/search_time)

                # child_designs.append(copy.deepcopy(search_new_vm_distribution_arr))
                # child_values.append(copy.deepcopy(search_objective_arr))

            # if local_fit2(search_objective_arr, weight_arr[member]) < local_fit2(population_value[member], weight_arr[member]):
            #     population[member] = copy.deepcopy(search_new_vm_distribution_arr)
            #     population_value[member] = copy.deepcopy(search_objective_arr)

        # population, population_value = update_population(child_designs, child_values, population, population_value, weight_arr)

        if i > 20:
        # if i > 0:
            non_member_list = list(range(population_size))
            for j in sorted(member_list, reverse = True):
                    non_member_list.pop(j)
            length_of_member_list = len(member_list)
            for j in range(length_of_member_list):
                parent_1 = member_list.pop()
                parent_2 = random.sample(non_member_list, k = 1)[0]
                
                parent_design_1 = population[parent_1]
                parent_design_2 = population[parent_2]
                
                child_design_candidate_list = GA(parent_design_1, parent_design_2)
                child_design_value_list = []
                for child_design in child_design_candidate_list:
                    search_leftover_resource_time_arr = update_leftover(child_design, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
                    search_cumulative_resource_time_arr, total_violation, total_invocation, next_vm_distribution_arr, slo_violation_arr = simulation(
                        vm_distribution_arr, child_design, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
                    search_objective_arr = update_objective(search_cumulative_resource_time_arr, total_violation, total_invocation, epoch)
                    child_design_value_list.append(search_objective_arr)

                population, population_value, update_ratio_table = update_population(
                    child_design_candidate_list, child_design_value_list, population, population_value, weight_arr, update_ratio_table)
                
    # for i in range(population_size):
    #     design = population[i]
    #     search_leftover_resource_time_arr = update_leftover(design, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
    #     search_cumulative_resource_time_arr, total_violation, total_invocation, next_vm_distribution_arr, slo_violation_arr = simulation(
    #         vm_distribution_arr, child_design, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
    #     joule_per_node = []
    #     for node_id in range(number_of_node):
    #         joule_in_15min = 0
    #         for time_step in range(900):
    #             joule_in_15min+=calculate_power(search_cumulative_resource_time_arr[node_id,time_step])
    #         joule_per_node.append(joule_in_15min)

    #     kwh_in_15min = sum(joule_per_node) * 0.0000002778
    #     energy_per_node = []
    #     for node_id in range(number_of_node):
    #         energy_per_node.append(joule_per_node[node_id] * 0.0000002778)
    #     hour = epoch//4
    #     hour = hour%24
    #     total_energy = kwh_in_15min * cop * cop_factor_list[hour]
    #     print(total_energy, population_value[i])
    
    print(epoch)
    np.savetxt('MO/e'+str(epoch)+'_values.txt', population_value, delimiter=',')
    for i in range(population_size):
        np.savetxt('MO/e'+str(epoch)+'_design'+str(i)+'.txt', population[i], delimiter=',')
    # exit()

    final_new_vm_distribution_arr = copy.deepcopy(search_new_vm_distribution_arr)
    # search_leftover_resource_time_arr = update_leftover(search_new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
    # search_cumulative_resource_time_arr, total_violation, total_invocation, next_vm_distribution_arr, slo_violation_arr = simulation(
    #     vm_distribution_arr, search_new_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
    # objective_arr = update_objective(search_cumulative_resource_time_arr, total_violation, total_invocation, epoch)
    # print(objective_arr)
    
    # return final_cumulative_resource_time_arr[:,:900], total_violation, total_invocation, final_cumulative_resource_time_arr[:,900:], next_vm_distribution_arr

def tri_optimizer(lut_list, func_distribution_arr, leftover_resource_time_arr, vm_distribution_arr, epoch):
    total_number_of_func_type = func_distribution_arr.shape[0]
    new_vm_distribution_arr = copy.deepcopy(vm_distribution_arr)
    # new_vm_distribution_arr[:,:]=0
    search_leftover_resource_time_arr=copy.deepcopy(leftover_resource_time_arr)
    
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*epoch_length)) ##two times of epoch length means considering left over resource usage for next epoch
    
    #record prewarm func id
    prewarm_func_list = []
    func_invocation_list = []
    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id,0]!=0:
            prewarm_func_list.append(func_id)
            func_invocation_list.append(func_distribution_arr[func_id,0])
        else:
            new_vm_distribution_arr[func_id,:]=0
    
    #record prewarm func id based on scoring
    scored_prewarm_func_list = []
    scored_invocation_count_list = []
    while len(func_invocation_list):
        max_invocation = max(func_invocation_list)
        max_invocation_index = func_invocation_list.index(max_invocation)
        scored_invocation_count_list.append(func_invocation_list.pop(max_invocation_index))
        scored_prewarm_func_list.append(prewarm_func_list.pop(max_invocation_index))
        
    func_priority_list = scored_prewarm_func_list
    search_leftover_resource_time_arr = update_leftover(new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
    search_cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
    
    
    black_list = [-1]
    search_history=[0]*total_number_of_func_type
    start_time = time.time()
    search_energy_cost, search_carbon = calculate_power_cost(search_cumulative_resource_time_arr, epoch)
    total_violation_count = copy.deepcopy(np.sum(slo_violation_arr[:,0]))
    total_carbon_emission = copy.deepcopy(search_carbon)
    while (time.time() - start_time) < 180:
        search_func_id = -1
        if (total_violation/total_invocation) > 0.05:
            search_func_id = random.choice(scored_prewarm_func_list)
            if search_func_id > -1:
                search_vm_distribution_arr = slo_optimization(new_vm_distribution_arr, search_func_id)
                search_leftover_resource_time_arr = update_leftover(search_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
                
                search_cumulative_resource_time_arr, total_violation, total_invocation, search_vm_distribution_arr, slo_violation_arr = simulation(
                    vm_distribution_arr, search_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
                if np.sum(slo_violation_arr[:,0]) < total_violation_count:
                    # print("SLO success", search_func_id, np.sum(slo_violation_arr[:,0])/total_invocation, total_violation_count/total_invocation)
                    total_violation_count = copy.deepcopy(np.sum(slo_violation_arr[:,0]))
                    new_vm_distribution_arr=copy.deepcopy(search_vm_distribution_arr)
                    search_energy_cost, search_carbon = calculate_power_cost(search_cumulative_resource_time_arr, epoch)
                    total_carbon_emission = copy.deepcopy(search_carbon)
        else:
            search_func_id = random.choice(scored_prewarm_func_list)
            if search_func_id > -1:
                search_vm_distribution_arr = carbon_optimization(new_vm_distribution_arr, search_func_id)
                search_leftover_resource_time_arr = update_leftover(search_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
                
                search_cumulative_resource_time_arr, search_total_violation, search_total_invocation, search_vm_distribution_arr, slo_violation_arr = simulation(
                    vm_distribution_arr, search_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
                search_energy_cost, search_carbon = calculate_power_cost(search_cumulative_resource_time_arr, epoch)
                if search_carbon < total_carbon_emission and (search_total_violation/search_total_invocation) <= 0.05:
                    # print('Carbon success', search_func_id, search_carbon, total_carbon_emission)
                    total_carbon_emission = copy.deepcopy(search_carbon)
                    total_violation_count = copy.deepcopy(np.sum(slo_violation_arr[:,0]))
                    new_vm_distribution_arr=copy.deepcopy(search_vm_distribution_arr)   
    
    # print('end search')
    leftover_resource_time_arr = update_leftover(new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
    cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, leftover_resource_time_arr, func_priority_list, lut_list)

    return cumulative_resource_time_arr[:,:900], total_violation, total_invocation, cumulative_resource_time_arr[:,900:], new_vm_distribution_arr

def single_step_slo_optimizer(lut_list, func_distribution_arr, leftover_resource_time_arr, vm_distribution_arr, new_vm_distribution_arr, epoch, step=1):
    total_number_of_func_type = func_distribution_arr.shape[0]
    search_leftover_resource_time_arr=copy.deepcopy(leftover_resource_time_arr)
    
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*epoch_length)) ##two times of epoch length means considering left over resource usage for next epoch
    
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
        
    func_priority_list = scored_prewarm_func_list
    search_leftover_resource_time_arr = update_leftover(new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
    cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
    
    black_list = [-1]
    search_history=[0]*total_number_of_func_type
    total_violation_count = copy.deepcopy(np.sum(slo_violation_arr[:,0]))
    for i in range(step):
        search_func_id = -1
        # for func_id in scored_prewarm_func_list:
        #     if func_id not in black_list:
        #         if slo_violation_arr[func_id,0] > violation_count:
        #             search_func_id = copy.deepcopy(func_id)
        #             violation_count = copy.deepcopy(slo_violation_arr[func_id,0])
        while search_func_id in black_list:
            search_func_id = random.choice(scored_prewarm_func_list)
        
        if search_func_id > -1:
        
            search_vm_distribution_arr = slo_optimization(new_vm_distribution_arr, search_func_id)
            
            search_leftover_resource_time_arr = update_leftover(search_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
            
            cumulative_resource_time_arr, total_violation, total_invocation, search_vm_distribution_arr, slo_violation_arr = simulation(
                vm_distribution_arr, search_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
            if np.sum(slo_violation_arr[:, 0]) < total_violation_count:
                # print("search success", search_func_id, func_distribution_arr[search_func_id, 0], new_vm_distribution_arr[search_func_id,:], violation_count, search_vm_distribution_arr[search_func_id,:], slo_violation_arr[search_func_id, 0])
                # search_history[search_func_id]=0
                total_violation_count = copy.deepcopy(np.sum(slo_violation_arr[:,0]))
                new_vm_distribution_arr=copy.deepcopy(search_vm_distribution_arr)
            else:
                # print("search fail", search_func_id, func_distribution_arr[search_func_id, 0], new_vm_distribution_arr[search_func_id,:], violation_count, search_vm_distribution_arr[search_func_id,:], slo_violation_arr[search_func_id, 0])
                search_history[search_func_id]+=1
                if search_history[search_func_id] > 10:
                    black_list.append(search_func_id)
        # print(i)
    
    # print('end search')
    search_leftover_resource_time_arr = update_leftover(new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
    cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
    
    energy_cost, carbon = calculate_power_cost(cumulative_resource_time_arr, epoch)

    return cumulative_resource_time_arr, total_violation, total_invocation, energy_cost, carbon, vm_distribution_arr

def multi_step_slo_optimizer(lut_list, func_distribution_arr, leftover_resource_time_arr, vm_distribution_arr, new_vm_distribution_arr, epoch, step=100):
    global first_interval_flag
    global this_interval_avail_resource_time
    global next_interval_avail_resource_time
    global number_of_node
    total_number_of_func_type = func_distribution_arr.shape[0]
    search_leftover_resource_time_arr=copy.deepcopy(leftover_resource_time_arr)
    
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*epoch_length)) ##two times of epoch length means considering left over resource usage for next epoch
    
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
        
    func_priority_list = scored_prewarm_func_list
    search_leftover_resource_time_arr = update_leftover(new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
    cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
    
    black_list = []
    search_history=[0]*total_number_of_func_type
    total_violation_count = copy.deepcopy(np.sum(slo_violation_arr[:,0]))
    for i in range(step):
        search_func_id = -1
        violation_count = -1
        for func_id in scored_prewarm_func_list:
            if func_id not in black_list:
                if slo_violation_arr[func_id,0] > violation_count:
                    search_func_id = copy.deepcopy(func_id)
                    violation_count = copy.deepcopy(slo_violation_arr[func_id,0])
        
        if search_func_id > -1:
        
            search_vm_distribution_arr = slo_optimization(new_vm_distribution_arr, search_func_id)
            
            search_leftover_resource_time_arr = update_leftover(search_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
            
            cumulative_resource_time_arr, total_violation, total_invocation, search_vm_distribution_arr, slo_violation_arr = simulation(
                vm_distribution_arr, search_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
            if np.sum(slo_violation_arr[:,0]) < total_violation_count:
                # print("search success", search_func_id, func_distribution_arr[search_func_id, 0], new_vm_distribution_arr[search_func_id,:], violation_count, search_vm_distribution_arr[search_func_id,:], slo_violation_arr[search_func_id, 0])
                # search_history[search_func_id]=0
                total_violation_count = copy.deepcopy(np.sum(slo_violation_arr[:,0]))
                new_vm_distribution_arr=copy.deepcopy(search_vm_distribution_arr)
            else:
                # print("search fail", search_func_id, func_distribution_arr[search_func_id, 0], new_vm_distribution_arr[search_func_id,:], violation_count, search_vm_distribution_arr[search_func_id,:], slo_violation_arr[search_func_id, 0])
                search_history[search_func_id]+=1
                if search_history[search_func_id] > 15:
                    black_list.append(search_func_id)
        else:
            break
        # print(i)
    
    # print('end search')
    search_leftover_resource_time_arr = update_leftover(new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
    cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
    
    energy_cost, carbon = calculate_power_cost(cumulative_resource_time_arr, epoch)

    return cumulative_resource_time_arr, total_violation, total_invocation, energy_cost, carbon, vm_distribution_arr

def single_step_carbon_optimizer(lut_list, func_distribution_arr, leftover_resource_time_arr, vm_distribution_arr, new_vm_distribution_arr, epoch, step=100):
    global first_interval_flag
    global this_interval_avail_resource_time
    global next_interval_avail_resource_time
    global number_of_node
    total_number_of_func_type = func_distribution_arr.shape[0]
    search_leftover_resource_time_arr=copy.deepcopy(leftover_resource_time_arr)
    
    cumulative_resource_time_arr = np.zeros((number_of_node, 2*epoch_length)) ##two times of epoch length means considering left over resource usage for next epoch
    
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
        
    func_priority_list = scored_prewarm_func_list
    search_leftover_resource_time_arr = update_leftover(new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
    cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
    energy_cost, carbon = calculate_power_cost(cumulative_resource_time_arr, epoch)
    
    black_list = [-1]
    search_history=[0]*total_number_of_func_type
    for i in range(step):
        search_func_id = -1
        violation_count = -1
        # for func_id in scored_prewarm_func_list:
        #     if func_id not in black_list:
        #         search_func_id = copy.deepcopy(func_id)
        while search_func_id in black_list:
            search_func_id = random.choice(scored_prewarm_func_list)
        
        if search_func_id > -1:
        
            search_vm_distribution_arr = carbon_optimization(new_vm_distribution_arr, search_func_id)
            
            search_leftover_resource_time_arr = update_leftover(search_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
            
            cumulative_resource_time_arr, total_violation, total_invocation, search_vm_distribution_arr, slo_violation_arr = simulation(
                vm_distribution_arr, search_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
            search_energy_cost, search_carbon = calculate_power_cost(cumulative_resource_time_arr, epoch)
            
            if search_carbon < carbon:
                # print("search success", search_func_id, func_distribution_arr[search_func_id, 0], new_vm_distribution_arr[search_func_id,:], violation_count, search_vm_distribution_arr[search_func_id,:], slo_violation_arr[search_func_id, 0])
                # search_history[search_func_id]=0
                new_vm_distribution_arr=copy.deepcopy(search_vm_distribution_arr)
                carbon = copy.deepcopy(search_carbon)
            else:
                # print("search fail", search_func_id, func_distribution_arr[search_func_id, 0], new_vm_distribution_arr[search_func_id,:], violation_count, search_vm_distribution_arr[search_func_id,:], slo_violation_arr[search_func_id, 0])
                search_history[search_func_id]+=1
                if search_history[search_func_id] > 5:
                    black_list.append(search_func_id)
        # print(i)
    
    # print('end search')
    search_leftover_resource_time_arr = update_leftover(new_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr)
    cumulative_resource_time_arr, total_violation, total_invocation, vm_distribution_arr, slo_violation_arr = simulation(
        vm_distribution_arr, new_vm_distribution_arr, func_distribution_arr, search_leftover_resource_time_arr, func_priority_list, lut_list)
    
    energy_cost, carbon = calculate_power_cost(cumulative_resource_time_arr, epoch)

    return cumulative_resource_time_arr, total_violation, total_invocation, energy_cost, carbon, vm_distribution_arr

def slo_optimization(vm_distribution_arr, search_func_id):
    search_vm_distribution_arr = copy.deepcopy(vm_distribution_arr)
    
    if search_vm_distribution_arr[search_func_id,0] == 0:
        container_location = random.randint(0,number_of_node-1)
        # container_location = 0
        search_vm_distribution_arr[search_func_id,container_location+1]=1
        search_vm_distribution_arr[search_func_id,0]=1
    else:
        if search_vm_distribution_arr[search_func_id,0] < number_of_node:
            if random.random() < 0.5:
                random.shuffle(search_vm_distribution_arr[search_func_id,1:])
            else:
                empty_location_list = np.where(search_vm_distribution_arr[search_func_id,1:] == 0)[0]
                container_location = random.choice(empty_location_list)
                search_vm_distribution_arr[search_func_id,container_location+1]=1
        else:
            container_location_list = np.where(search_vm_distribution_arr[search_func_id,1:] == 1)[0]
            container_location = random.choice(container_location_list)
            search_vm_distribution_arr[search_func_id,container_location+1]=0
            
    search_vm_distribution_arr[search_func_id,0]=np.count_nonzero(search_vm_distribution_arr[search_func_id,1:])        
    return search_vm_distribution_arr

def local_search(vm_distribution_arr, search_func_id):
    search_vm_distribution_arr = copy.deepcopy(vm_distribution_arr)
    
    # print(search_vm_distribution_arr[search_func_id,1:])

    # if random.random() < 0.5:
    #     while (search_vm_distribution_arr[search_func_id,1:] == vm_distribution_arr[search_func_id,1:]).all():
    #         random.shuffle(search_vm_distribution_arr[search_func_id,1:])
    #     # print(search_func_id, 'shuffule', search_vm_distribution_arr[search_func_id,1:])
    if search_vm_distribution_arr[search_func_id,0] == 0:
        empty_location_list = np.where(search_vm_distribution_arr[search_func_id,1:] == 0)[0]
        container_location = random.choice(empty_location_list)
        search_vm_distribution_arr[search_func_id,container_location+1]=1
        # print(search_func_id, 'add', search_vm_distribution_arr[search_func_id,1:])
    else:
        if search_vm_distribution_arr[search_func_id,0] == number_of_node:
            container_location_list = np.where(search_vm_distribution_arr[search_func_id,1:] == 1)[0]
            container_location = random.choice(container_location_list)
            search_vm_distribution_arr[search_func_id,container_location+1]=0
            # print(search_func_id, 'remove', search_vm_distribution_arr[search_func_id,1:])
        else:
            if random.random() < 0.5:
                while (search_vm_distribution_arr[search_func_id,1:] == vm_distribution_arr[search_func_id,1:]).all():
                    random.shuffle(search_vm_distribution_arr[search_func_id,1:])
            else:
                if random.random() < 0.5:
                    empty_location_list = np.where(search_vm_distribution_arr[search_func_id,1:] == 0)[0]
                    container_location = random.choice(empty_location_list)
                    search_vm_distribution_arr[search_func_id,container_location+1]=1
                    # print(search_func_id, 'add', search_vm_distribution_arr[search_func_id,1:])
                else:
                    container_location_list = np.where(search_vm_distribution_arr[search_func_id,1:] == 1)[0]
                    container_location = random.choice(container_location_list)
                    search_vm_distribution_arr[search_func_id,container_location+1]=0
                    # print(search_func_id, 'remove', search_vm_distribution_arr[search_func_id,1:])
            
    search_vm_distribution_arr[search_func_id,0]=np.count_nonzero(search_vm_distribution_arr[search_func_id,1:])        
    return search_vm_distribution_arr

def carbon_optimization(vm_distribution_arr, search_func_id):
    search_vm_distribution_arr = copy.deepcopy(vm_distribution_arr)
    
    if search_vm_distribution_arr[search_func_id,0] == 0:
        # container_location = random.randint(0,number_of_node-1)
        # search_vm_distribution_arr[search_func_id,container_location+1]=1
        # search_vm_distribution_arr[search_func_id,0]=1
        pass
    else:
        if random.random() < 0.5:
            random.shuffle(search_vm_distribution_arr[search_func_id,1:])
            # search_vm_distribution_arr = distribute_container(search_vm_distribution_arr)
        else:
            non_empty_location_list = np.where(search_vm_distribution_arr[search_func_id,1:] == 1)[0]
            container_location = random.choice(non_empty_location_list)
            search_vm_distribution_arr[search_func_id,container_location+1]=0
            
    search_vm_distribution_arr[search_func_id,0]=np.count_nonzero(search_vm_distribution_arr[search_func_id,1:])        
    return search_vm_distribution_arr

def cost_optimization(vm_distribution_arr, search_func_id):
    search_vm_distribution_arr = copy.deepcopy(vm_distribution_arr)
    
    if search_vm_distribution_arr[search_func_id,0] == 0:
        container_location = random.randint(0,number_of_node-1)
        # container_location = 0
        search_vm_distribution_arr[search_func_id,container_location+1]=1
        search_vm_distribution_arr[search_func_id,0]=1
    else:
        if search_vm_distribution_arr[search_func_id,0] < number_of_node:
            if random.random() < 0.5:
                random.shuffle(search_vm_distribution_arr[search_func_id,1:])
            else:
                empty_location_list = np.where(search_vm_distribution_arr[search_func_id,1:] == 0)[0]
                container_location = random.choice(empty_location_list)
                search_vm_distribution_arr[search_func_id,container_location+1]=1
        else:
            container_location_list = np.where(search_vm_distribution_arr[search_func_id,1:] == 1)[0]
            container_location = random.choice(container_location_list)
            search_vm_distribution_arr[search_func_id,container_location+1]=0
            
    search_vm_distribution_arr[search_func_id,0]=np.count_nonzero(search_vm_distribution_arr[search_func_id,1:])        
    return search_vm_distribution_arr

def update_leftover(search_vm_distribution_arr, vm_distribution_arr, scored_prewarm_func_list, leftover_resource_time_arr):
    new_leftover_resource_time_arr = copy.deepcopy(leftover_resource_time_arr)
    
    for func_id in scored_prewarm_func_list:
        for node_id in range(number_of_node):
            if search_vm_distribution_arr[func_id,node_id+1]:
                if vm_distribution_arr[func_id,node_id+1] == 0:
                    new_leftover_resource_time_arr[node_id,:epoch_length]+=idle_res_usage
            else:
                if vm_distribution_arr[func_id,node_id+1]:
                    new_leftover_resource_time_arr[node_id,:vm_shutdown_time]+=idle_res_usage
    
    return new_leftover_resource_time_arr

def calculate_power(core_usage):
    if core_usage == 0:
        power = 0
    else:
        power = 0.000002*math.pow(core_usage,4) - 0.0003*math.pow(core_usage,3) - 0.0184*math.pow(core_usage,2) + 5.1778*core_usage + 128.42
    return power

def calculate_power_cost(resource_usage_arr, i):
    joule_in_15min = 0
    for node_id in range(number_of_node):
        for time_step in range(epoch_length):
            joule_in_15min+=calculate_power(resource_usage_arr[node_id,time_step])
    kwh_in_15min = joule_in_15min * 0.0000002778
    hour = i//4
    hour = hour%24
    total_energy = kwh_in_15min * cop * cop_factor_list[hour]
    total_energy_cost = total_energy * price_list[hour]

    carbon = total_energy * carbon_density_list[hour]
    
    return total_energy_cost, carbon

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--laxity', type=int, help='deadline laxity', default=10)
    parser.add_argument('-s', '--slo', type=float, help='slo contraint', default=0.05)
    parser.add_argument('-e', '--epoch', type=int, help='number of epoch', default=96)
    parser.add_argument('-t', '--time', type=int, help='decision time', default=180)
    parser.add_argument('-n', '--node', type=int, help='number of nodes', default=4)
    parser.add_argument('-d', '--duration', type=int, help='times of duration', default=1)
    parser.add_argument('-r', '--request', type=int, help='times of request', default=1)
    parser.add_argument('-f', '--framework', type=str, help='framework', default='DSLO', choices= 
                        ['SLO', 'Load', 'Ideal', 'Back', 'Hybrid', 'Score', 'Binary', 'Mscore', 'DSLO', 'Qtrain', 'Qtest', 'Search'])
    args = parser.parse_args()

    ddl_laxity = args.laxity
    slo_constraint = args.slo
    number_of_epoch = args.epoch
    decision_time = args.time
    number_of_node = args.node
    time_of_duration = args.duration
    time_of_request = args.request
    framework = args.framework
    print(number_of_node, time_of_duration, time_of_request, framework)

    
    
    a_week_func_distribution_list =[] #(func_id, total_rate, rate_in_900sec)
    total_func_list=list(range(424))

    with open("a_week_func_distribution_list_original", "rb") as f:   #Unpickling
        a_week_func_distribution_list = pickle.load(f)
    print("distribution load success")

    lut_list = []
    with open("a_week_func_runtime_list_original", "rb") as f:   #Unpickling
        lut_list = pickle.load(f)
    lut_arr = np.array(lut_list, dtype=int)
    print("runtime load success")
    # print(lut_arr)
    
    objective_arr = np.zeros((672,3))
    leftover_resource_time_arr = np.zeros((number_of_node, 900))
    vm_distribution_arr = np.zeros((424, number_of_node+1)) ## (number of container per ID, location 0, ..., location n) as one line, 424 lines represent 424 IDs in Azure trace

    time_limit = 60
    regular_vm_list = []
    for i in range(424):
        if lut_list[i] > time_limit:
            regular_vm_list.append(i)
    if framework == 'Qtrain':

        # Define the environment 0%, 20% 40% 60% 80%
        n_states = pow(4,number_of_node+1)  # Number of states in the grid world, [req_portion, node_0_loal_level, node_1_loal_level, ...]
        n_actions = pow(2,number_of_node) - 1 # Number of possible actions (up, down, left, right)
        goal_state = 0  # Goal state

        # Initialize Q-table with zeros
        Q_table = np.zeros((n_states, n_actions), dtype=np.half)
        # Q_table = np.zeros((n_states, n_actions), dtype=None)

        # Define parameters
        learning_rate = 0.8
        discount_factor = 0.95
        exploration_prob = 0.2
    
    if framework == 'Qtest':
        n_states = pow(4,number_of_node+1)  # Number of states in the grid world, [req_portion, node_0_loal_level, node_1_loal_level, ...]
        n_actions = pow(2,number_of_node) - 1 # Number of possible actions (up, down, left, right)
        goal_state = 0  # Goal state

        # Initialize Q-table with zeros
        Q_table = np.zeros((n_states, n_actions), dtype=np.half)

        # Define parameters
        learning_rate = 0.8
        discount_factor = 0.95
        exploration_prob = 0.2

        with open("q_table_n"+str(number_of_node), "rb") as f:   #Pickling
            Q_table = pickle.load(f)
    
    if framework == 'Search':
        
        # per_node_cooling_efficiency = np.arange(0.6, 0.9, 0.3/number_of_node).tolist()
        number_of_population = 20
        number_of_objective = 3 
        ## loading weights
        with open('data/pop'+str(number_of_population)+'_obj'+str(number_of_objective)+'_weight.csv', newline='') as f:
            reader = csv.reader(f)
            weight_arr = list(reader)
        for i in range(len(weight_arr)):
            weight_arr[i] = [float(j) for j in weight_arr[i]]

        with open('data/pop'+str(number_of_population)+'_obj'+str(number_of_objective)+'_weight_n.csv', newline='') as f:
            reader = csv.reader(f)
            neighbor_arr = list(reader)
        for i in range(len(weight_arr)):
            neighbor_arr[i] = [int(j) for j in neighbor_arr[i]]

    
    # for i in range(672):
    for i in range(0,number_of_epoch): 
    # for i in range(11,12):
        s_time = time.time()
        previous_func_type = [] 
        if framework == 'Qtrain':
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr, vm_distribution_arr = q_training(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[i], leftover_resource_time_arr, vm_distribution_arr, Q_table, learning_rate, discount_factor, exploration_prob, n_actions)           
        if framework == 'Qtest':
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr, vm_distribution_arr = q_testing(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[i], leftover_resource_time_arr, vm_distribution_arr, Q_table, learning_rate, discount_factor, exploration_prob, n_actions)           
        # if framework == 'SLO':
        #     resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr, vm_distribution_arr = violation_minimum_policy(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[i], leftover_resource_time_arr, vm_distribution_arr)
        if framework == 'Load':
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr, vm_distribution_arr = load_minimum_policy(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[i], leftover_resource_time_arr, vm_distribution_arr)
        if framework == 'Ideal':
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr = ideal_policy(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[i], leftover_resource_time_arr)
        if framework == 'Score':
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr, vm_distribution_arr = score_policy(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[i], leftover_resource_time_arr, vm_distribution_arr)
        if framework == 'Mscore':
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr, vm_distribution_arr = multi_score_policy(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[i], leftover_resource_time_arr, vm_distribution_arr)
        if framework == 'SLO':
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr, vm_distribution_arr = slo_optimizer(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[i], leftover_resource_time_arr, vm_distribution_arr, decision_time)
        if framework == 'DSLO':
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr, vm_distribution_arr = dual_optimizer(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[i], leftover_resource_time_arr, vm_distribution_arr, i, decision_time, slo_constraint)
        if framework == 'Search':
            hybrid_search(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[i], leftover_resource_time_arr, vm_distribution_arr, i, decision_time, slo_constraint, weight_arr)
        if framework == 'TSLO':
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr, vm_distribution_arr = tri_optimizer(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[i], leftover_resource_time_arr, vm_distribution_arr, i)
        if framework == 'Binary':
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr, vm_distribution_arr = binary_search(time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[i], leftover_resource_time_arr, vm_distribution_arr, i)
        if framework == 'Hybrid':
            node_id = 0
            regular_vm_location = []
            for j in range(len(regular_vm_list)):
                leftover_resource_time_arr[node_id,:]+=2
                regular_vm_location.append(node_id)
                node_id+=1
                node_id%=number_of_node
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr, vm_distribution_arr  = multi_hybrid_policy(
                time_of_duration*lut_arr, time_of_request*a_week_func_distribution_list[i], leftover_resource_time_arr, vm_distribution_arr, regular_vm_list, regular_vm_location)
        if framework == 'Back':
            resource_usage_arr = np.zeros((number_of_node, 900))
            violation_count = 0
        
        if framework != 'Search':
            single_epoch_objective_arr = update_objective(resource_usage_arr, violation_count, invocation_count, i)
            objective_arr[i,:] = single_epoch_objective_arr
            print(i, objective_arr[i,:])

    print("violation_rate (%), carbon (g), water (L)")
    ave_violation_rate = np.mean(objective_arr[:i+1,0])
    cumulative_carbon = np.sum(objective_arr[:i+1,1])
    cumulative_water = np.sum(objective_arr[:i+1,2])/100
    print(ave_violation_rate, cumulative_carbon, cumulative_water)

    if framework == 'Qtrain':
        with open("q_table_n"+str(number_of_node), "wb") as f:   #Pickling
            pickle.dump(Q_table, f)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if framework == 'DSLO' or framework == 'SLO':
        with open('outputs/'+framework+'_l'+str(ddl_laxity)+'_n'+str(number_of_node)+'_d'+str(time_of_duration)+'_r'+str(time_of_request)+'_e'+str(number_of_epoch)+'_t'+str(decision_time)+'_s'+str(slo_constraint*100), 'w') as f:
            f.writelines(str(np.sum(objective_arr[:,0])) + ',' + str(np.sum(objective_arr[:,1])/np.sum(objective_arr[:,2])) + ',' + str(np.sum(objective_arr[:,3])) + ',' + str(np.mean(objective_arr[:i,4])))
    else:
        if framework != 'Search':
            with open('outputs/'+framework+'_l'+str(ddl_laxity)+'_n'+str(number_of_node)+'_d'+str(time_of_duration)+'_r'+str(time_of_request)+'_e'+str(number_of_epoch), 'w') as f:
                f.writelines(str(ave_violation_rate) + ',' + str(cumulative_carbon) + ',' + str(cumulative_water))

        
    
    