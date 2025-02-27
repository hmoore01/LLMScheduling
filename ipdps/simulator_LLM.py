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
import pandas as pd
import MultiAgentRL


random.seed(10)
# np.random.seed(10)
# per func type: 2000 
# container_startup_time = 20 #sec
# container_showdown_time = 15 #sec
# container_idle_resource = 100 #100mi CPU
# container_startup_resource = 300 #100mi CPU
# number_of_node = 4
# epoch_length = 900
# real_time_node_load_arr = np.zeros((number_of_node, epoch_length))
# number_of_core_per_node = 256
# max_cpu_time = number_of_node*number_of_core_per_node*1000
# this_interval_avail_resource_time = np.ones(epoch_length,dtype=float)*max_cpu_time
# next_interval_avail_resource_time = np.ones(epoch_length,dtype=float)*max_cpu_time
# resource_per_request = 2
ddl_laxity = 10
# resource = 0.1
# per_node_cooling_efficiency = []
global_min_weight = 0.0005
first_interval_flag = 1

# container_startup_time = 0 #sec
# container_showdown_time = 0 #sec
# container_idle_resource = 0 #100mi CPU
# container_startup_resource = 0 #100mi CPU

a100_startup_time = 15
h100_startup_time = 10
a100_idle_resource = 60
h100_idle_resource = 50
a100_startup_resource = 400
h100_startup_resource = 350
number_of_a100_nodes = 4
number_of_h100_nodes = 4
total_number_of_nodes = number_of_a100_nodes + number_of_h100_nodes
epoch_length = 900
real_time_node_load_arr = np.zeros((total_number_of_nodes, epoch_length))
number_of_core_per_a100_node = 512
number_of_core_per_h100_node = 768
max_gpu_time_a100 = number_of_a100_nodes * number_of_core_per_a100_node * 1000
max_gpu_time_h100 = number_of_h100_nodes * number_of_core_per_h100_node * 1000
max_gpu_time = max_gpu_time_a100 + max_gpu_time_h100
this_interval_avail_resource_time = np.ones(epoch_length, dtype=float) * max_gpu_time
next_interval_avail_resource_time = np.ones(epoch_length, dtype=float) * max_gpu_time
resource_per_request_a100 = 8
resource_per_request_h100 = 6
vm_startup_time = 0
vm_shutdown_time = 30

idle_res_usage_a100 = 4
idle_res_usage_h100 = 3

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


def simulation(vm_distribution_arr,
               new_vm_distribution_arr,
               func_distribution_arr,
               leftover_resource_time_arr,
               func_priority_list,
               lut_list):

    total_number_of_func_type = func_distribution_arr.shape[0]

    # slo_violation_arr = (number_of_functions, 2)
    #   - [:, 0] => SLO violation counts
    #   - [:, 1] => total invocations processed
    slo_violation_arr = np.zeros((total_number_of_func_type, 2))
    last_epoch_func_list = []

    # Prewarm function list
    scored_prewarm_func_list = func_priority_list

    # Initialize cumulative resource tracking
    # We track (2 * epoch_length) + 1 in case leftover extends
    cumulative_resource_time_arr = np.zeros((total_number_of_nodes, (2 * epoch_length) + 1))
    cumulative_resource_time_arr[:, :epoch_length + 1] += leftover_resource_time_arr

    # Shutdown unnecessary VMs based on arrival prediction
    for func_id in range(total_number_of_func_type):
        if func_id not in scored_prewarm_func_list:
            previous_prewarm_location_list = list(np.nonzero(vm_distribution_arr[func_id, 1:])[0])
            for node in previous_prewarm_location_list:
                # Add idle resource usage for the shutdown period
                if node >= number_of_a100_nodes:
                    # This is an H100 node
                    cumulative_resource_time_arr[node, :vm_shutdown_time] += h100_idle_resource
                else:
                    # This is an A100 node
                    cumulative_resource_time_arr[node, :vm_shutdown_time] += a100_idle_resource

    # Calculate per-node startup time (depends on number of containers/VMs)
    vm_startup_time_list = []
    number_of_container_per_node = np.count_nonzero(new_vm_distribution_arr[:, 1:], axis=0)
    for node_id in range(total_number_of_nodes):
        number_of_colocated_containers = number_of_container_per_node[node_id]
        startup_time = (
            h100_startup_time if node_id >= number_of_a100_nodes else a100_startup_time
        )
        vm_startup_time_list.append(startup_time * number_of_colocated_containers)

    # Deep copy the new VM distribution
    vm_distribution_arr = copy.deepcopy(new_vm_distribution_arr)

    # Simulation loop
    for node_id in range(total_number_of_nodes):
        queued_func_id_list = []
        queued_func_req_list = []
        queued_func_delay_list = []

        for second_time in range(epoch_length):
            # A) Process queued functions
            updated_queued_func_id_list = []
            updated_queued_func_req_list = []
            updated_queued_func_delay_list = []

            for i in range(len(queued_func_id_list)):
                func_id = queued_func_id_list[i]
                number_of_invocation = queued_func_req_list[i]
                delay_time = queued_func_delay_list[i]

                # Only process if startup time for that node has passed
                if second_time >= vm_startup_time_list[node_id]:
                    load_list = [
                        max(cumulative_resource_time_arr[nn, second_time:second_time + 15])
                        for nn in range(total_number_of_nodes)
                    ]
                    # This version of calculate_resource_time_v5 returns 5 values
                    (
                        resource_time_arr,
                        slo_violation_count,
                        disused_cold_startup,
                        delay_time,
                        water_usage
                    ) = calculate_resource_time_v5(
                        load_list, lut_list, func_id, number_of_invocation, node_id,
                        delay_time, node_type='H100' if node_id >= number_of_a100_nodes else 'A100'
                    )

                    # *** INCREMENT THE TOTAL INVOCATIONS HERE ***
                    slo_violation_arr[func_id, 1] += number_of_invocation

                    # Update resource usage
                    cumulative_resource_time_arr[:, second_time:second_time + 15] += resource_time_arr
                    # Update SLO violations
                    slo_violation_arr[func_id, 0] += slo_violation_count

                    # If leftover requests remain, re-queue them
                    if disused_cold_startup > 0:
                        updated_queued_func_id_list.append(func_id)
                        updated_queued_func_req_list.append(disused_cold_startup)
                        updated_queued_func_delay_list.append(delay_time)

            # Update queued requests
            queued_func_id_list = copy.deepcopy(updated_queued_func_id_list)
            queued_func_req_list = copy.deepcopy(updated_queued_func_req_list)
            queued_func_delay_list = copy.deepcopy(updated_queued_func_delay_list)

            # B) Process new arrivals
            #    (arrivals for functions that were pre-warmed, i.e., in func_priority_list)
            for func_id in scored_prewarm_func_list:
                number_of_invocation = func_distribution_arr[func_id, second_time]

                if number_of_invocation > 0 and second_time >= vm_startup_time_list[node_id]:
                    load_list = [
                        max(cumulative_resource_time_arr[nn, second_time:second_time + 15])
                        for nn in range(total_number_of_nodes)
                    ]
                    (
                        resource_time_arr,
                        slo_violation_count,
                        disused_cold_startup,
                        delay_time,
                        water_usage
                    ) = calculate_resource_time_v5(
                        load_list, lut_list, func_id, number_of_invocation, node_id, 0,
                        node_type='H100' if node_id >= number_of_a100_nodes else 'A100'
                    )

                    # *** INCREMENT THE TOTAL INVOCATIONS HERE AS WELL ***
                    slo_violation_arr[func_id, 1] += number_of_invocation

                    # Accumulate resource usage
                    cumulative_resource_time_arr[:, second_time:second_time + 15] += resource_time_arr
                    # Add any SLO violations
                    slo_violation_arr[func_id, 0] += slo_violation_count

        # C) Handle leftover queued requests after the entire epoch
        for i in range(len(queued_func_id_list)):
            the_func_id = queued_func_id_list[i]
            # We treat leftover as all violations if it never got processed
            slo_violation_count = queued_func_req_list[i]
            # No new invocations here, so no increment to column 1
            # (but if you want to count them as arrived, do so)
            slo_violation_arr[the_func_id, 0] += slo_violation_count

    # D) Update leftover resources for the next epoch
    leftover_resource_time_arr[:, :epoch_length + 1] = cumulative_resource_time_arr[:, epoch_length:]

    # E) Summarize the results
    total_violation = np.sum(slo_violation_arr[:, 0])
    total_invocation = np.sum(slo_violation_arr[:, 1])

    return cumulative_resource_time_arr, int(total_violation), int(total_invocation), vm_distribution_arr, slo_violation_arr


def split_request(number_of_invocation, node_id, vm_distribution_arr):
    # Determine total available VMs
    total_vms = np.sum(vm_distribution_arr[:, 1:])

    if total_vms == 0:
        # No available VMs, return all requests unprocessed
        return 0

    # Calculate the proportion of invocations this node can handle
    node_vms = np.sum(vm_distribution_arr[:, node_id + 1])
    proportional_share = (node_vms / total_vms) if total_vms > 0 else 0

    # Allocate invocations based on proportional share
    allocated_invocations = int(number_of_invocation * proportional_share)

    return allocated_invocations


def calculate_resource_time_v5(load_list, lut_list, func_id, number_of_invocation, node_id, delay_time, node_type=None):
    node_properties = {
        'A100': {
            'max_capacity': 512,  # Max tokens per second
            'resource_per_token': 8,  # Resource units per token
            'startup_time': 15,  # Startup time in seconds
            'carbon_rate': 0.02,  # CO2 emissions per resource unit
            'energy_per_token': 0.0003,  # kWh used per resource token
        },
        'H100': {
            'max_capacity': 768,
            'resource_per_token': 6,
            'startup_time': 10,
            'carbon_rate': 0.015,
            'energy_per_token': 0.00025,  # kWh used per resource token
        }
    }

    if node_type not in node_properties:
        raise ValueError(f"Invalid node type: {node_type}")

    # Extract node properties
    max_capacity = node_properties[node_type]['max_capacity']
    resource_per_token = node_properties[node_type]['resource_per_token']
    carbon_rate = node_properties[node_type]['carbon_rate']
    energy_per_token = node_properties[node_type]['energy_per_token']

    # Water density in liters/kWh
    water_density = 0.53

    # Prepare output structures
    resource_time_arr = np.zeros(15)
    number_of_invocation = int(number_of_invocation)  # ensure integer

    # Compute total resource requirement and available capacity
    total_required_resources = number_of_invocation * resource_per_token
    available_resources = max_capacity - load_list[node_id]

    # Initialize return variables
    disused_cold_startup = 0
    carbon_emissions = 0.0
    water_usage = 0.0
    time_to_first_token = delay_time

    if available_resources >= total_required_resources:
        # All invocations can be served
        resource_time_arr[:number_of_invocation] = resource_per_token

        # Emissions & water usage for ALL allocated resources
        carbon_emissions = total_required_resources * carbon_rate
        # Convert resource -> kWh, then multiply by water density
        water_usage = (total_required_resources * energy_per_token) * water_density

    else:
        # Partial service (some are unprocessed)
        allocatable_invocations = int(available_resources // resource_per_token)
        allocatable_invocations = max(0, allocatable_invocations)

        disused_cold_startup = number_of_invocation - allocatable_invocations
        resource_time_arr[:allocatable_invocations] = resource_per_token

        # Calculate resource usage for the portion we can serve
        allocated_resource_units = allocatable_invocations * resource_per_token
        carbon_emissions = allocated_resource_units * carbon_rate
        water_usage = (allocated_resource_units * energy_per_token) * water_density

        # If not all requests can be processed, add +1 to the time-to-first-token
        time_to_first_token = delay_time + 1

    return resource_time_arr, time_to_first_token, disused_cold_startup, carbon_emissions, water_usage


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


def dual_optimizer(duration, request, leftover_resource_time_arr, vm_distribution_arr, epoch, decision_time,
                   slo_constraint, llama_8b_requests, llama_70b_requests, node_properties):

    # --------------------------------------------------------------------
    #   Configuration / Setup
    # --------------------------------------------------------------------
    print("\n[DEBUG] Entering dual_optimizer() for epoch:", epoch)
    print(f"[DEBUG] duration={duration}, request={request}, decision_time={decision_time}, slo_constraint={slo_constraint}")
    print(f"[DEBUG] llama_8b_requests={llama_8b_requests}, llama_70b_requests={llama_70b_requests}")

    # Assume you have some global or module-level variables for node counts, e.g.:
    total_number_of_nodes = number_of_a100_nodes + number_of_h100_nodes
    print(f"[DEBUG] total_number_of_nodes = {total_number_of_nodes}")

    # Convert to numpy arrays if not already
    vm_distribution_arr = np.asarray(vm_distribution_arr)
    leftover_resource_time_arr = np.asarray(leftover_resource_time_arr)

    # Validate shapes (example checks)
    if vm_distribution_arr.shape != (total_number_of_nodes, EPOCH_LENGTH + 1):
        print(f"[WARN] Invalid vm_distribution_arr shape: {vm_distribution_arr.shape}, reinitializing.")
        vm_distribution_arr = np.zeros((total_number_of_nodes, EPOCH_LENGTH + 1))

    if leftover_resource_time_arr.shape != (total_number_of_nodes, EPOCH_LENGTH + 1):
        print(f"[WARN] Invalid leftover_resource_time_arr shape: {leftover_resource_time_arr.shape}, reinitializing.")
        leftover_resource_time_arr = np.zeros((total_number_of_nodes, EPOCH_LENGTH + 1))

    # Convert requests to int if > 0, else 0
    llama_8b_requests = int(llama_8b_requests) if llama_8b_requests > 0 else 0
    llama_70b_requests = int(llama_70b_requests) if llama_70b_requests > 0 else 0
    print(f"[DEBUG] Final count: Llama_8B={llama_8b_requests}, Llama_70B={llama_70b_requests}")

    # --------------------------------------------------------------------
    #   Build a function distribution array
    # --------------------------------------------------------------------
    func_distribution_arr = np.zeros((len(node_properties), EPOCH_LENGTH + 1))
    func_distribution_arr[0, :] = llama_8b_requests
    func_distribution_arr[1, :] = llama_70b_requests

    # --------------------------------------------------------------------
    #   Initialize arrays for simulation
    # --------------------------------------------------------------------
    new_vm_distribution_arr = np.zeros_like(vm_distribution_arr)
    search_leftover_resource_time_arr = np.zeros_like(leftover_resource_time_arr)
    cumulative_resource_time_arr = np.zeros((total_number_of_nodes, 2 * (EPOCH_LENGTH + 1)))

    print(f"[DEBUG] vm_distribution_arr shape: {vm_distribution_arr.shape}")
    print(f"[DEBUG] leftover_resource_time_arr shape: {leftover_resource_time_arr.shape}")
    print(f"[DEBUG] new_vm_distribution_arr shape: {new_vm_distribution_arr.shape}")
    print(f"[DEBUG] search_leftover_resource_time_arr shape: {search_leftover_resource_time_arr.shape}")
    print(f"[DEBUG] cumulative_resource_time_arr shape: {cumulative_resource_time_arr.shape}")

    # --------------------------------------------------------------------
    #   Prepare Prewarm Function List
    # --------------------------------------------------------------------
    prewarm_func_list = []
    func_invocation_list = []

    if llama_8b_requests > 0:
        prewarm_func_list.append(0)  # Mapped ID for 'Llama_8B'
        func_invocation_list.append(llama_8b_requests)
    if llama_70b_requests > 0:
        prewarm_func_list.append(1)  # Mapped ID for 'Llama_70B'
        func_invocation_list.append(llama_70b_requests)

    print(f"[DEBUG] prewarm_func_list: {prewarm_func_list}, func_invocation_list: {func_invocation_list}")

    # Sort functions by invocation count (descending)
    scored_prewarm_func_list = []
    scored_invocation_count_list = []
    while func_invocation_list:
        max_invocation = max(func_invocation_list)
        max_invocation_index = func_invocation_list.index(max_invocation)
        scored_invocation_count_list.append(func_invocation_list.pop(max_invocation_index))
        scored_prewarm_func_list.append(prewarm_func_list.pop(max_invocation_index))

    func_priority_list = scored_prewarm_func_list
    print(f"[DEBUG] Sorted func_priority_list: {func_priority_list}, invocation counts: {scored_invocation_count_list}")

    # --------------------------------------------------------------------
    #   Update leftover resources (based on initial distribution)
    # --------------------------------------------------------------------
    search_leftover_resource_time_arr = update_leftover(
        new_vm_distribution_arr,  # next epoch distribution
        vm_distribution_arr,      # current epoch distribution
        scored_prewarm_func_list,
        leftover_resource_time_arr
    )
    print(f"[DEBUG] leftover updated with update_leftover()")

    # Validate shape consistency before running simulation
    if vm_distribution_arr.shape != leftover_resource_time_arr.shape:
        raise ValueError(f"[ERROR] Shape mismatch before simulation: "
                         f"vm_distribution_arr {vm_distribution_arr.shape}, "
                         f"leftover_resource_time_arr {leftover_resource_time_arr.shape}")

    # --------------------------------------------------------------------
    #   Initial Simulation
    # --------------------------------------------------------------------
    print("[DEBUG] Starting initial simulation...")
    (
        search_cumulative_resource_time_arr,
        total_violation,
        total_invocation,
        vm_distribution_arr,
        slo_violation_arr
    ) = simulation(
        vm_distribution_arr,
        new_vm_distribution_arr,
        func_distribution_arr,
        search_leftover_resource_time_arr,
        func_priority_list,
        node_properties
    )

    print(f"[DEBUG] Initial simulation done. total_violation={total_violation}, total_invocation={total_invocation}")
    print("[DEBUG] SLO violation array:\n", slo_violation_arr)

    # Calculate initial power/cost
    search_energy_cost, search_carbon = calculate_power_cost(search_cumulative_resource_time_arr, epoch)
    total_violation_count = np.sum(slo_violation_arr[:, 0])
    total_carbon_emission = search_carbon

    print(f"[DEBUG] Initial power cost: energy={search_energy_cost}, carbon={search_carbon}")
    print(f"[DEBUG] total_violation_count={total_violation_count}, total_carbon_emission={total_carbon_emission}")

    # --------------------------------------------------------------------
    #   Main Optimization Loop with 60-Second Time Limit
    # --------------------------------------------------------------------
    time_limit = 60  # seconds per epoch
    start_time = time.time()

    while (time.time() - start_time) < time_limit:
        # Safely compute ratio to avoid ZeroDivisionError
        if total_invocation == 0:
            ratio = 0.0
            print(f"[WARN] total_invocation is 0; ratio set to 0.0")
        else:
            ratio = total_violation / total_invocation

        print(f"[DEBUG] ratio={ratio:.4f}, slo_constraint={slo_constraint}")

        search_func_id = -1
        if ratio > slo_constraint:
            # -------------------------------------------------------
            # SLO optimization
            # -------------------------------------------------------
            print("[DEBUG] ratio > slo_constraint => Attempt SLO optimization")
            if scored_prewarm_func_list:
                search_func_id = random.choice(scored_prewarm_func_list)
                print(f"[DEBUG] SLO optimization with func_id={search_func_id}")

                search_vm_distribution_arr = slo_optimization(new_vm_distribution_arr, search_func_id)
                search_leftover_resource_time_arr = update_leftover(
                    search_vm_distribution_arr,
                    vm_distribution_arr,
                    scored_prewarm_func_list,
                    leftover_resource_time_arr
                )

                try:
                    (
                        search_cumulative_resource_time_arr,
                        total_violation,
                        total_invocation,
                        search_vm_distribution_arr,
                        slo_violation_arr
                    ) = simulation(
                        vm_distribution_arr,
                        search_vm_distribution_arr,
                        func_distribution_arr,
                        search_leftover_resource_time_arr,
                        func_priority_list,
                        node_properties
                    )
                    print(f"[DEBUG] SLO optimization simulation => violation={total_violation}, invocation={total_invocation}")
                except Exception as e:
                    print(f"[ERROR] Simulation error (SLO optimization) func_id={search_func_id}: {e}")
                    continue

                # If SLO violation improved, update distribution
                if np.sum(slo_violation_arr[:, 0]) < total_violation_count:
                    total_violation_count = np.sum(slo_violation_arr[:, 0])
                    new_vm_distribution_arr = copy.deepcopy(search_vm_distribution_arr)
                    search_energy_cost, search_carbon = calculate_power_cost(search_cumulative_resource_time_arr, epoch)
                    total_carbon_emission = search_carbon
                    print("[DEBUG] SLO improved; updated new_vm_distribution_arr & carbon_emission.")
        else:
            # -------------------------------------------------------
            # Carbon optimization
            # -------------------------------------------------------
            print("[DEBUG] ratio <= slo_constraint => Attempt Carbon optimization")
            if scored_prewarm_func_list:
                search_func_id = random.choice(scored_prewarm_func_list)
                print(f"[DEBUG] Carbon optimization with func_id={search_func_id}")

                search_vm_distribution_arr = carbon_optimization(new_vm_distribution_arr, search_func_id)
                search_leftover_resource_time_arr = update_leftover(
                    search_vm_distribution_arr,
                    vm_distribution_arr,
                    scored_prewarm_func_list,
                    leftover_resource_time_arr
                )

                try:
                    (
                        search_cumulative_resource_time_arr,
                        search_total_violation,
                        search_total_invocation,
                        search_vm_distribution_arr,
                        slo_violation_arr
                    ) = simulation(
                        vm_distribution_arr,
                        search_vm_distribution_arr,
                        func_distribution_arr,
                        search_leftover_resource_time_arr,
                        func_priority_list,
                        node_properties
                    )
                    print(f"[DEBUG] Carbon optimization simulation => violation={search_total_violation}, invocation={search_total_invocation}")
                except Exception as e:
                    print(f"[ERROR] Simulation error (Carbon optimization) func_id={search_func_id}: {e}")
                    continue

                search_energy_cost, search_carbon = calculate_power_cost(search_cumulative_resource_time_arr, epoch)
                ratio_new = (search_total_violation / search_total_invocation) if search_total_invocation > 0 else float('inf')
                print(f"[DEBUG] ratio_new={ratio_new:.4f}, search_carbon={search_carbon:.2f}, total_carbon_emission={total_carbon_emission:.2f}")

                # If carbon improved AND we still meet SLO
                if (search_carbon < total_carbon_emission) and (ratio_new <= slo_constraint):
                    total_carbon_emission = search_carbon
                    total_violation_count = np.sum(slo_violation_arr[:, 0])
                    new_vm_distribution_arr = copy.deepcopy(search_vm_distribution_arr)
                    print("[DEBUG] Carbon improved; updated new_vm_distribution_arr & total_violation_count.")

    # --------------------------------------------------------------------
    #   Final update after the 60-second search
    # --------------------------------------------------------------------
    print("[DEBUG] Exiting optimization loop (60s limit reached). Final simulation...")

    leftover_resource_time_arr = update_leftover(
        new_vm_distribution_arr,
        vm_distribution_arr,
        scored_prewarm_func_list,
        leftover_resource_time_arr
    )
    (
        cumulative_resource_time_arr,
        total_violation,
        total_invocation,
        vm_distribution_arr,
        slo_violation_arr
    ) = simulation(
        vm_distribution_arr,
        new_vm_distribution_arr,
        func_distribution_arr,
        leftover_resource_time_arr,
        func_priority_list,
        node_properties
    )

    print("[DEBUG] Final simulation done.")
    print(f"[DEBUG] final total_violation={total_violation}, total_invocation={total_invocation}")

    return (cumulative_resource_time_arr[:, :duration],
            total_violation,
            total_invocation,
            cumulative_resource_time_arr[:, duration:],
            new_vm_distribution_arr)




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

    # 1) Violation Rate (%)
    if total_invocation == 0:
        print(f"Warning: total_invocation is zero at epoch {epoch}. Setting objective_arr[0] to 0.")
        objective_arr[0] = 0
    else:
        objective_arr[0] = (total_violation / total_invocation) * 100

    # 2) Calculate energy usage for each node by iterating over actual array shape
    joule_per_node = []
    # Instead of EPOCH_LENGTH+1, we get the second dimension from the array
    max_time = search_cumulative_resource_time_arr.shape[1]

    for node_id in range(number_of_node):
        joule_in_15min = 0
        # Accumulate power for each timestep that actually exists
        for time_step in range(max_time):
            joule_in_15min += calculate_power(search_cumulative_resource_time_arr[node_id, time_step])
        joule_per_node.append(joule_in_15min)

    # Convert Joules -> kWh
    kwh_in_15min = sum(joule_per_node) * 0.0000002778  # 1 Joule = 2.7778e-7 kWh
    energy_per_node = [joule * 0.0000002778 for joule in joule_per_node]

    # 3) Determine hour for carbon/water calculations
    hour = epoch // 4
    hour = hour % 24

    # 4) Calculate total energy with cooling factor
    total_energy = kwh_in_15min * cop * cop_factor_list[hour]

    # 5) Carbon calculation (grams)
    carbon = total_energy * carbon_density_list[hour]

    # 6) Water usage
    # a) Electricity -> water usage
    electricity_water = total_energy * water_density

    # b) Evaporation water
    water_latent_heat = 0.66    # kWh / L
    cycle_of_concentration = 5

    # Node-specific cooling efficiency
    per_node_cooling_efficiency = []
    for node_id in range(number_of_node):
        if energy_per_node[node_id] > 0.09:
            per_node_cooling_efficiency.append(0.3)
        elif energy_per_node[node_id] > 0.06:
            per_node_cooling_efficiency.append(0.6)
        else:
            per_node_cooling_efficiency.append(0.9)

    # c) Base evaporation formula
    #   (But see the next step for an "adjusted" version)
    evaporation_water = sum(
        energy / cooling_eff / water_latent_heat
        for energy, cooling_eff in zip(energy_per_node, per_node_cooling_efficiency)
    )

    # Adjust overall cooling efficiency by total energy
    if total_energy > 0.4:
        cooling_efficiency = 0.9
    elif total_energy > 0.3:
        cooling_efficiency = 0.6
    else:
        cooling_efficiency = 0.9

    # Overwrite evaporation_water with new formula
    evaporation_water = total_energy / water_latent_heat / cooling_efficiency

    # d) Blowdown water
    blowdown_water = evaporation_water / (cycle_of_concentration - 1)

    # 7) Populate objective array
    objective_arr[1] = carbon
    objective_arr[2] = (electricity_water + evaporation_water + blowdown_water) * 100

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
        # Ensure func_id is valid for indexing
        if not isinstance(func_id, int):
            try:
                func_id = int(func_id)
            except ValueError:
                print(f"Invalid func_id encountered: {func_id}")
                continue

        for node_id in range(total_number_of_nodes):  # Using total_number_of_nodes from global variables
            if node_id < number_of_a100_nodes:
                idle_resource = a100_idle_resource
                startup_time = a100_startup_time
            else:
                idle_resource = h100_idle_resource
                startup_time = h100_startup_time

            # Perform bounds checks before accessing array elements
            if node_id + 1 < search_vm_distribution_arr.shape[1]:
                try:
                    if search_vm_distribution_arr[func_id, node_id + 1]:
                        if vm_distribution_arr[func_id, node_id + 1] == 0:
                            new_leftover_resource_time_arr[node_id, :epoch_length] += idle_resource
                    else:
                        if vm_distribution_arr[func_id, node_id + 1]:
                            new_leftover_resource_time_arr[node_id, :vm_shutdown_time] += idle_resource
                except IndexError as e:
                    print(f"IndexError in update_leftover for func_id {func_id}, node_id {node_id}: {e}")
                    print(f"search_vm_distribution_arr shape: {search_vm_distribution_arr.shape}")
                    print(f"vm_distribution_arr shape: {vm_distribution_arr.shape}")
                    print(f"leftover_resource_time_arr shape: {new_leftover_resource_time_arr.shape}")
                    continue

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
    parser.add_argument('-s', '--slo', type=float, help='SLO constraint', default=0.05)
    parser.add_argument('-e', '--epoch', type=int, help='number of epochs', default=256)
    parser.add_argument('-t', '--time', type=int, help='decision time', default=180)
    parser.add_argument('-n', '--node', type=int, help='number of nodes', default=8)
    parser.add_argument('-d', '--duration', type=int, help='duration time', default=1)
    parser.add_argument('-r', '--request', type=int, help='number of requests', default=1)
    parser.add_argument('-f', '--framework', type=str, help='framework', default='SARL_Train', choices=[
                        'SLO', 'Load', 'Ideal', 'Back', 'Hybrid', 'Score', 'Binary',
                        'Mscore', 'DSLO', 'Qtrain', 'Qtest', 'Search', 'MARL', 'Helix',
                        'SARL_Train', 'SARL_Eval', 'Splitwise', 'Swarm'])
    args = parser.parse_args()

    # Load and process the trace file
    trace = pd.read_csv('BurstGPT_processed.csv')
    func_id_mapping = {'Llama_8B': 0, 'Llama_70B': 1}
    trace['Mapped Func ID'] = trace.apply(
        lambda row: 0 if row['Llama_8B Requests'] > 0 else
        1 if row['Llama_70B Requests'] > 0 else
        -1,
        axis=1
    )
    max_epoch = trace['Epoch'].max()
    grouped_trace = trace.groupby('Epoch')

    # Constants
    EPOCH_LENGTH = 900

    # Arguments
    ddl_laxity = args.laxity
    slo_constraint = args.slo
    number_of_epoch = args.epoch
    decision_time = args.time
    number_of_node = args.node
    time_of_duration = args.duration
    time_of_request = args.request
    framework = args.framework
    print(f"Initialized with {number_of_node} nodes, {time_of_duration} duration, {time_of_request} requests, framework: {framework}")


    node_properties = {
        'Llama_8B': {
            'resource_per_request': 10,
            'completion_time_factor': 1.0
        },
        'Llama_70B': {
            'resource_per_request': 20,
            'completion_time_factor': 1.5
        }
    }

    # Initialize arrays with consistent shapes
    objective_arr = np.zeros((number_of_epoch, 3))
    leftover_resource_time_arr = np.zeros((number_of_node, EPOCH_LENGTH + 1))
    vm_distribution_arr = np.zeros((EPOCH_LENGTH + 1, number_of_node))
    func_distribution_arr = np.zeros((len(func_id_mapping), EPOCH_LENGTH + 1))  # For invocation counts

    # Populate func_distribution_arr with invocation data
    for epoch_idx in range(number_of_epoch):
        if epoch_idx in grouped_trace.groups:
            epoch_data = grouped_trace.get_group(epoch_idx)
            func_distribution_arr[0, epoch_idx] = epoch_data['Llama_8B Requests'].sum()
            func_distribution_arr[1, epoch_idx] = epoch_data['Llama_70B Requests'].sum()


    training_split_epoch = 0.7 * 32
    training_RL = True
    eval_RL = False
    # Debugging outputs
    print(f"Objective array shape: {objective_arr.shape}")
    print(f"Leftover resource time array shape: {leftover_resource_time_arr.shape}")
    print(f"VM distribution array shape: {vm_distribution_arr.shape}")
    print(f"Function distribution array shape: {func_distribution_arr.shape}")
    print(f"First epoch invocations: {func_distribution_arr[:, 0]}")
    cumulative_carbon = 0
    cumulative_water = 0
    avg_ttft = 0
    time_limit = 60
    regular_vm_list = []
    epoch_counter = 0

    for epoch_idx in range(65,128):
        epoch_counter = epoch_counter + 1
        llama_8b_requests = 0
        llama_70b_requests = 0

        if epoch_idx in grouped_trace.groups:
            epoch_data = grouped_trace.get_group(epoch_idx)
            llama_8b_requests = epoch_data['Llama_8B Requests'].sum()
            llama_70b_requests = epoch_data['Llama_70B Requests'].sum()
            print(llama_70b_requests)
            print(llama_8b_requests)

        if framework == 'DSLO':
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr, vm_distribution_arr = dual_optimizer(
                time_of_duration, time_of_request, leftover_resource_time_arr, vm_distribution_arr, epoch_idx,
                decision_time, slo_constraint,
                llama_8b_requests, llama_70b_requests, node_properties
            )
        elif framework == 'Helix':
            import Helix
            resource_usage_arr, total_carbon_emissions, total_water_usage, invocation_count, leftover_resource_time_arr, vm_distribution_arr, average_time_to_first_token = Helix.milp_optimizer(
                time_of_duration, time_of_request, leftover_resource_time_arr, vm_distribution_arr, epoch_idx,
                decision_time, slo_constraint,
                llama_8b_requests, llama_70b_requests, node_properties
            )
            print(f"Total Carbon Emissions: {total_carbon_emissions}")
            print(f"Total Water Usage: {total_water_usage}")
            print(f"Total Invocation Served: {invocation_count}")
            cumulative_carbon+= total_carbon_emissions
            cumulative_water+= total_water_usage
            avg_ttft+= average_time_to_first_token
        elif framework == 'Splitwise':
            import Splitwise

            resource_usage_arr, total_carbon_emissions, total_water_usage, invocation_count, leftover_resource_time_arr, vm_distribution_arr, average_time_to_first_token = Splitwise.splitwise(
                time_of_duration, time_of_request, leftover_resource_time_arr, vm_distribution_arr, epoch_idx,
                decision_time, slo_constraint, func_distribution_arr,
                llama_8b_requests, llama_70b_requests, node_properties
            )
            print(f"Total Carbon Emissions: {total_carbon_emissions}")
            print(f"Total Water Usage: {total_water_usage}")
            print(f"Total Invocation Served: {invocation_count}")
            cumulative_carbon += total_carbon_emissions
            cumulative_water += total_water_usage
            avg_ttft += average_time_to_first_token
        elif framework == 'SARL_Train':
            import SingleAgentRL

            config = {
                "leftover_resource_time_arr": leftover_resource_time_arr,
                "vm_distribution_arr": vm_distribution_arr,
                "node_properties": node_properties,
                "number_of_nodes": number_of_node,
                "number_of_epoch": number_of_epoch,
                "time_of_duration": time_of_duration,
                "time_of_request": time_of_request,
                "decision_time": decision_time,
                "slo_constraint": slo_constraint,
                "epoch_idx": epoch_idx,
                "func_distribution_arr": func_distribution_arr,
                "max_steps": 20
            }
            env = SingleAgentRL.ResourceEnv(config)
            trained_model = SingleAgentRL.train_single_agent(env, total_timesteps=1024)

        elif framework == 'SARL_Eval':
            import SingleAgentRL

            config = {
                "leftover_resource_time_arr": leftover_resource_time_arr,
                "vm_distribution_arr": vm_distribution_arr,
                "node_properties": node_properties,
                "number_of_nodes": number_of_node,
                "number_of_epoch": number_of_epoch,
                "time_of_duration": time_of_duration,
                "time_of_request": time_of_request,
                "decision_time": decision_time,
                "slo_constraint": slo_constraint,
                "epoch_idx": epoch_idx,
                "func_distribution_arr": func_distribution_arr,
                "max_steps": 1
            }
            env = SingleAgentRL.ResourceEnv(config)
            results = SingleAgentRL.evaluate_agent(
                        model_path="trained_models/single_agent/ppo_single_agent.zip",
                        env_config=config,
                        n_eval_episodes=1,
                        save_results=True,
                        results_path="LLM_Results/eval_results.txt"
                        )
            cumulative_carbon += results["total_carbon"]
            cumulative_water += results["total_water"]
            avg_ttft += results["average_time_token"]

        elif framework == 'Qtrain':
            n_states = pow(4, number_of_node + 1)
            n_actions = pow(2, number_of_node) - 1
            Q_table = np.zeros((n_states, n_actions), dtype=np.half)
            learning_rate = 0.8
            discount_factor = 0.95
            exploration_prob = 0.2
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr, vm_distribution_arr = q_training(
                time_of_duration, time_of_request, leftover_resource_time_arr, vm_distribution_arr, Q_table,
                learning_rate, discount_factor, exploration_prob, n_actions
            )
        elif framework == 'Qtest':
            n_states = pow(4, number_of_node + 1)
            n_actions = pow(2, number_of_node) - 1
            with open(f"q_table_n{number_of_node}", "rb") as f:
                Q_table = pickle.load(f)
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr, vm_distribution_arr = q_testing(
                time_of_duration, time_of_request, leftover_resource_time_arr, vm_distribution_arr, Q_table,
                0.8, 0.95, 0.2, n_actions
            )
        elif framework == 'Load':
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr, vm_distribution_arr = load_minimum_policy(
                time_of_duration, time_of_request, leftover_resource_time_arr, vm_distribution_arr
            )
        elif framework == 'Ideal':
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr = ideal_policy(
                time_of_duration, time_of_request, leftover_resource_time_arr
            )
        elif framework == 'Score':
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr, vm_distribution_arr = score_policy(
                time_of_duration, time_of_request, leftover_resource_time_arr, vm_distribution_arr
            )
        elif framework == 'Mscore':
            resource_usage_arr, violation_count, invocation_count, leftover_resource_time_arr, vm_distribution_arr = multi_score_policy(
                time_of_duration, time_of_request, leftover_resource_time_arr, vm_distribution_arr
            )
        elif framework == 'Search':
            hybrid_search(
                time_of_duration, time_of_request, leftover_resource_time_arr, vm_distribution_arr, epoch_idx,
                decision_time, slo_constraint
            )
        elif framework == 'MARL':
            config = {
                "leftover_resource_time_arr": leftover_resource_time_arr,
                "vm_distribution_arr": vm_distribution_arr,
                "node_properties": node_properties,
                "number_of_nodes": number_of_node,
                "number_of_epoch": number_of_epoch,
                "time_of_duration": time_of_duration,
                "time_of_request": time_of_request,
                "decision_time": decision_time,
                "slo_constraint": slo_constraint,
                "epoch_idx": epoch_idx,
                "func_distribution_arr": func_distribution_arr,
                "max_steps": 10
            }

            print(epoch_idx)

            env = MultiAgentRL.ResourceEnv(config)

            if epoch_idx < training_split_epoch:
                models = MultiAgentRL.train_marl_agents(env, total_timesteps=10000)

            if epoch_idx >= training_split_epoch:
                from stable_baselines3 import PPO

                save_dir = "trained_models"
                agent_names = ["carbon_model", "time_model", "water_model"]
                env_classes = {
                    "time_model": TimeEnv,
                    "carbon_model": CarbonEnv,
                    "water_model": WaterEnv
                }

                def load_trained_models(save_dir, agent_names):
                    models = {}
                    for agent_name in agent_names:
                        model_path = os.path.join(save_dir, f"{agent_name}.zip")
                        if os.path.exists(model_path):
                            print(f"Loading model for {agent_name} from {model_path}...")
                            models[agent_name] = PPO.load(model_path)
                        else:
                            raise FileNotFoundError(f"Model for {agent_name} not found in {save_dir}.")
                    return models

                models = load_trained_models(save_dir, agent_names)


                def evaluate_agent(agent, env_class, config, eval_episodes=1):
                    env = env_class(config)
                    results = {}

                    for _ in range(eval_episodes):
                        obs, info = env.reset()
                        done = False
                        while not done:
                            action, _ = agent.predict(obs, deterministic=True)
                            print("Action:",action)
                            obs, reward, terminated, truncated, info = env.step(action)
                            done = terminated or truncated

                            # Record all metrics dynamically from the environment's info
                            for key, value in info.items():
                                if key not in results:
                                    results[key] = []
                                results[key].append(value)

                    # Compute the mean for each metric
                    env.close()
                    return {key: np.mean(values) for key, values in results.items()}


                def evaluate_all_agents(models, config, env_classes, eval_episodes=1):
                    agent_results = {}
                    for agent_name, env_class in env_classes.items():
                        print(f"Evaluating {agent_name}...")
                        agent_results[agent_name] = evaluate_agent(models[agent_name], env_class, config, eval_episodes)
                    return agent_results


                def log_evaluation_results(results, log_file=None):
                    for agent, metrics in results.items():
                        print(f"{agent}:")
                        for metric, value in metrics.items():
                            print(f"  {metric}: {value:.4f}")

                    # Optionally log to a file
                    if log_file:
                        with open(log_file, "w") as f:
                            for agent, metrics in results.items():
                                f.write(f"{agent}:\n")
                                for metric, value in metrics.items():
                                    f.write(f"  {metric}: {value:.4f}\n")


                # Evaluate models
                results = evaluate_all_agents(models, config, env_classes, eval_episodes=1)

                log_evaluation_results(results, log_file="evaluation_results.txt")

        else:
            resource_usage_arr = np.zeros((number_of_node, EPOCH_LENGTH + 1))
            violation_count = 0

        if epoch_idx < training_split_epoch:
            resource_usage_arr = np.zeros((number_of_node, EPOCH_LENGTH + 1))
            violation_count = 0
            invocation_count = 0
            single_epoch_objective_arr = update_objective(resource_usage_arr, violation_count, invocation_count, epoch_idx)
            objective_arr[epoch_idx, :] = single_epoch_objective_arr
            print(epoch_idx, objective_arr[epoch_idx, :])
        if epoch_idx >= training_split_epoch:
            resource_usage_arr = np.zeros((number_of_node, EPOCH_LENGTH + 1))
            violation_count = 0
            invocation_count = 0
            single_epoch_objective_arr = update_objective(resource_usage_arr, violation_count, invocation_count,
                                                          epoch_idx)
            objective_arr[epoch_idx, :] = single_epoch_objective_arr
            print(epoch_idx, objective_arr[epoch_idx, :])

    # Final results
    # ave_violation_rate = np.mean(objective_arr[:, 0])
    # cumulative_carbon = np.sum(objective_arr[:, 1])
    # cumulative_water = np.sum(objective_arr[:, 2]) / 100
    print(f"Average Time to first Token (s): {avg_ttft/epoch_counter}, Carbon (g): {cumulative_carbon}, Water (L): {cumulative_water}")

    # Save results
    if framework == 'Qtrain':
        with open(f"q_table_n{number_of_node}", "wb") as f:
            pickle.dump(Q_table, f)

    if not os.path.exists('LLL_Results'):
        os.makedirs('LLL_Results')
    output_path = f'LLM_Results/{framework}_l{ddl_laxity}_n{number_of_node}_d{time_of_duration}_r{time_of_request}_e{number_of_epoch}'
    with open(output_path, 'w') as f:
        f.writelines(f"Number of 15 minute epochs: {epoch_counter}, Average Time to First Token (s): {avg_ttft/epoch_counter},Cumulative Carbon Emissions (g): {cumulative_carbon},Cumulative Water Usage (L): {cumulative_water}")
