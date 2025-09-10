import numpy as np
import math
import random
import copy
import csv
# import hvwfg
import geo_simulator
numbe_of_region = 4
epoch_length = 900
ddl_laxity = 10
# ddl_laxity = []
# for func_id in range(424):
#     ddl_laxity.append(random.randint(2,10))
resource_per_request = 2
number_of_node_per_region = 1000
number_of_core_per_region = 256 * number_of_node_per_region


cop_factor_list = [0.85, 0.92, 1.02, 1.10, 1.14, 1.20, 1.21, 1.22, 
                    1.21, 1.20, 1.15, 1.11, 1.03, 0.96, 0.91, 0.816, 
                    0.83, 0.82, 0.82, 0.81, 0.81, 0.80, 0.80, 0.80]

price_list = [[48,48,48,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39],
              [29,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,20,20,29,29,29],
              [15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,15.45,21,21,21,21,15.45],
              [38.5,38.5,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,38.5,38.5,38.5,38.5]]

carbon_list = [0.000229, 0.000548, 0.000477, 0.000227] # LA, DEN, DTE, NY
water_list = [0.19, 1.93, 1.89, 3.22]

network_penalty_arr = np.array([[0.00,0.05,0.25,0.75],
                                [0.05,0.00,0.05,0.25],
                                [0.25,0.05,0.00,0.05],
                                [0.75,0.25,0.05,0.00]])

utility_list = []

for func_id in range(424):
    utility_list.append([random.randint(1,9)*0.1,-random.randint(1,9)*0.1])
# print(utility_list[0])

def simulation(lut_list, func_distribution_arr, leftover_resource_time_arr, original_req_distribution_arr, req_distribution_arr, new_req_distribution_arr, epoch):
    total_number_of_func_type = func_distribution_arr.shape[0]
    cumulative_resource_time_arr = np.zeros((numbe_of_region, 2*epoch_length)) ##two times of epoch length means considering left over resource usage for next epoch
    cumulative_resource_time_arr[:,:epoch_length] += leftover_resource_time_arr 
    split_func_distribution_arr = split(func_distribution_arr, new_req_distribution_arr)
    slo_violation_arr = np.zeros((total_number_of_func_type, 2))

    scored_prewarm_func_list=[]
    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id,0]:
            scored_prewarm_func_list.append(func_id)
    
    for region_id in range(numbe_of_region):
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
                # if second_time >= vm_startup_time_list[container_location]:
                if second_time >= 0:
                    load_list=[]
                    for node_location in range(numbe_of_region):
                        load_list.append(max(cumulative_resource_time_arr[node_location,second_time:second_time+15]))
                    resource_time_arr, slo_violation_count, disused_cold_startup, delay_time = calculate_resource_time(load_list, lut_list, func_id, number_of_invocation, region_id, delay_time)
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
       
            ## update queue list
            queued_func_id_list = copy.deepcopy(updated_queued_func_id_list)
            # print(queued_func_id_list)
            queued_func_req_list = copy.deepcopy(updated_queued_func_req_list)
            # print(queued_func_req_list)
            queued_func_delay_list = copy.deepcopy(updated_queued_func_delay_list)
            # print(queued_func_delay_list)

            ## process just-come-in func
            for i in range(len(scored_prewarm_func_list)):
                func_id = scored_prewarm_func_list[i]
                splitted_number_of_invocation = split_func_distribution_arr[region_id, func_id, second_time+1]
                container_location = region_id
                # print('yes')
                if splitted_number_of_invocation:
                    slo_violation_arr[func_id,1]+=splitted_number_of_invocation

                    load_list=[]
                    for node_location in range(numbe_of_region):
                        load_list.append(max(cumulative_resource_time_arr[node_location,second_time:second_time+15]))
                    resource_time_arr, slo_violation_count, disused_cold_startup, delay_time = calculate_resource_time(load_list, lut_list, func_id, splitted_number_of_invocation, region_id, 0)
                    cumulative_resource_time_arr[:,second_time:second_time+900]+=resource_time_arr
                    slo_violation_arr[func_id,0]+=slo_violation_count
                    if disused_cold_startup:
                        if delay_time > 0:
                            queued_func_id_list.append(func_id)
                            queued_func_req_list.append(disused_cold_startup)
                            queued_func_delay_list.append(delay_time)
                        else:
                            print("Warning!!!")
        
        for i in range(len(queued_func_id_list)):
            the_func_id = queued_func_id_list[i]
            slo_violation_count = queued_func_req_list[i]
            slo_violation_arr[the_func_id,0] += slo_violation_count

    # load_list = []
    # for i in range(numbe_of_region):
    #     load_list.append(np.mean(cumulative_resource_time_arr[i,:])/number_of_core_per_region)
    # print(load_list)
    
    updated_slo_violation_arr = update_slo(slo_violation_arr, original_req_distribution_arr, new_req_distribution_arr)
    # print(updated_slo_violation_arr)
    utility = calculate_utility(updated_slo_violation_arr, new_req_distribution_arr, utility_list)
    slo_rate = np.sum(updated_slo_violation_arr[:,0]) / np.sum(updated_slo_violation_arr[:,1]) 
    energy_per_region = calculate_energy(cumulative_resource_time_arr, epoch)
    cost_per_region = calculate_cost(energy_per_region, epoch)
    carbon_per_region = calculate_carbon(energy_per_region, epoch)
    water_per_region = calculate_water(energy_per_region, epoch)
    cost = sum(cost_per_region)
    carbon = sum(carbon_per_region)
    water = sum(water_per_region)
    
    epoch_objectives = [utility/1000000, cost/1000, carbon, water/1000]        
    return epoch_objectives, cumulative_resource_time_arr[:,900:]

def calculate_utility(updated_slo_violation_arr, new_req_distribution_arr, utility_list):
    utility = 0
    total_number_of_func_type = new_req_distribution_arr.shape[0]
    for func_id in range(total_number_of_func_type):
        if updated_slo_violation_arr[func_id,1]:
            number_of_success = updated_slo_violation_arr[func_id,1] - updated_slo_violation_arr[func_id,0]
            number_of_fail = updated_slo_violation_arr[func_id,0]
            
            utility+=number_of_success*utility_list[func_id][0]
            utility+=number_of_fail*utility_list[func_id][1]
            # print(number_of_success, number_of_fail, utili)
    return utility

def update_slo(slo_violation_arr, original_req_distribution_arr, new_req_distribution_arr):
    total_number_of_func_type = new_req_distribution_arr.shape[0]
    new_slo_violation_arr = copy.deepcopy(slo_violation_arr)
    
    for func_id in range(total_number_of_func_type):
        if slo_violation_arr[func_id,1]:
            region_req_flow = np.zeros((numbe_of_region,numbe_of_region),dtype=float)
            the_index = np.where(new_req_distribution_arr[:,0] == func_id)[0][0]
            difference = new_req_distribution_arr[the_index,1:] - original_req_distribution_arr
            # print(difference)
            # exit()
            for region_id in range(numbe_of_region):
                if region_id == 0:
                    search_list = [1,2,3]
                if region_id == 1:
                    search_list = [0,2,3]
                    # search_list = [2,0,3]
                if region_id == 2:
                    search_list = [1,3,0]
                    # search_list = [3,1,0]
                if region_id == 3:
                    search_list = [2,1,0]

                if difference[region_id] >0 : # move_in exists
                    for move_out_region in search_list:
                        if difference[move_out_region] < 0:
                            the_difference = difference[move_out_region] + difference[region_id]
                            if the_difference > 0: # move_out < move_in
                                region_req_flow[move_out_region,region_id]= -difference[move_out_region]
                                # print(region_req_flow, move_out_region, region_id, difference[region_id])
                                # exit()
                                difference[region_id] = the_difference
                                difference[move_out_region] = 0
                            else: # move_out >= move_in
                                # if move_out_region == 3 and region_id == 1:
                                #     print('here')
                                #     exit(difference[region_id])
                                region_req_flow[move_out_region,region_id]= difference[region_id]
                                difference[region_id] = 0
                                difference[move_out_region] = the_difference

            add_on_slo = np.sum(region_req_flow*network_penalty_arr)
            # print(region_req_flow, add_on_slo)
            
            new_slo_violation_arr[func_id,0]+=(slo_violation_arr[func_id,1]-slo_violation_arr[func_id,0])*add_on_slo
            # print(slo_violation_arr[func_id,0], new_slo_violation_arr[func_id,0], slo_violation_arr[func_id,1])


    return new_slo_violation_arr

def calculate_water(energy_per_region, epoch):
    water_per_region = []
    hour = epoch//4
    hour = hour%24
    for region_id in range(numbe_of_region):
        water_per_region.append(energy_per_region[region_id]*water_list[region_id])

    return water_per_region

    return water_per_region
def calculate_carbon(energy_per_region, epoch):
    carbon_per_region=[]
    hour = epoch//4
    hour = hour%24
    for region_id in range(numbe_of_region):
        carbon_per_region.append(energy_per_region[region_id]*carbon_list[region_id])

    return carbon_per_region


def calculate_cost(energy_per_region, epoch):
    cost_per_region = []
    hour = epoch//4
    hour = hour%24
    for region_id in range(numbe_of_region):
        cost_per_region.append(energy_per_region[region_id]*price_list[region_id][hour])

    return cost_per_region

def calculate_cost(energy_per_region, epoch):
    cost_per_region = []
    hour = epoch//4
    hour = hour%24
    for region_id in range(numbe_of_region):
        cost_per_region.append(energy_per_region[region_id]*price_list[region_id][hour])

    return cost_per_region

def calculate_power(core_usage):
    # if core_usage == 0:
    if 0:
        power = 0
    else:
        power = 0.000002*math.pow(core_usage,4) - 0.0003*math.pow(core_usage,3) - 0.0184*math.pow(core_usage,2) + 5.1778*core_usage + 128.42
    return power

def calculate_energy(cumulative_resource_time_arr, epoch):
    load_per_region = cumulative_resource_time_arr / number_of_node_per_region
    energy_per_region = []
    hour = epoch//4
    hour = hour%24
    
    for region_id in range(numbe_of_region):
        hour+=1*region_id
        hour = hour%24
        cop = cop_factor_list[hour] * 1.5
        joule_in_15min = 0
        for time_step in range(900):
            joule_in_15min+=calculate_power(load_per_region[region_id,time_step])
        energy_per_region.append(joule_in_15min * 0.0000002778 * number_of_node_per_region)
    return energy_per_region

def calculate_resource_time(nodes_load_list, lut_list, func_id, number_of_invocation, region_id, delay_time): 
    # print('ddl_laxity', ddl_laxity)
    node_load = nodes_load_list[region_id]
    base_execution_time = lut_list[func_id] #baseline execution time
    ddl_time = base_execution_time * ddl_laxity
    
    resource = resource_per_request #core CPU
    
    resource_time_arr = np.zeros((numbe_of_region, 900))
    slo_violation_count = 0

    prewarm_shortage = number_of_invocation

    ##check the ddl before execution
    if delay_time + base_execution_time > ddl_time and prewarm_shortage:
        delay_time=-1
        slo_violation_count = number_of_invocation
        disused_cold_startup = 0       
        prewarm_shortage = 0

    if prewarm_shortage:
        if (node_load + resource * prewarm_shortage) <= number_of_core_per_region:
            resource_time_arr[region_id, :base_execution_time] += resource * prewarm_shortage
            node_load += resource * prewarm_shortage
            prewarm_shortage = 0
            disused_cold_startup = 0
        else:
            survived_cold_startup = (number_of_core_per_region - node_load) // resource
            disused_cold_startup = prewarm_shortage - survived_cold_startup
            resource_time_arr[region_id, :base_execution_time] += resource * survived_cold_startup
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

def split(func_distribution_arr, new_req_distribution_arr):
    total_number_of_func_type = func_distribution_arr.shape[0]
    split_func_distribution_arr = np.zeros((numbe_of_region, total_number_of_func_type, epoch_length+1), dtype=int)
    for func_id in range(total_number_of_func_type):
        if func_distribution_arr[func_id, 0]:
            if func_id in new_req_distribution_arr[:,0]:
                the_index = np.where(new_req_distribution_arr[:,0] == func_id)[0][0]
                total_req_num = func_distribution_arr[func_id, 0]
                for region_id in range(numbe_of_region):
                    split_func_distribution_arr[region_id, func_id, 0] = math.floor(total_req_num*new_req_distribution_arr[the_index,region_id+1])

                difference = total_req_num - np.sum(split_func_distribution_arr[:, func_id, 0])
                if difference < 0:
                    print(func_id, region_id, "warning")
                    print(new_req_distribution_arr[the_index,:])
                    for j in range(numbe_of_region):
                        print(math.floor(total_req_num*new_req_distribution_arr[the_index,j+1]))
                    exit()
                split_func_distribution_arr[0, func_id, 0] += difference

                for region_id in range(numbe_of_region):
                    list_length=epoch_length
                    list_sum=split_func_distribution_arr[region_id, func_id, 0]
                    # rand_n = [ random.random() for i in range(list_length) ]
                    rand_n = [ 1 for i in range(list_length) ]
                    result = [ math.floor(i * list_sum / sum(rand_n)) for i in rand_n ] 
                    while list_sum - sum(result): 
                        difference = list_sum - sum(result)
                        result[random.randint(0,list_length-1)] += random.randint(0, difference)
                    split_func_distribution_arr[region_id, func_id, 1:] = result

            else:
                print('func id', func_id, 'is not split by policy' )
                exit()

    return split_func_distribution_arr
# total_number_of_func_type = 424
# # np.random.seed(10)
# # per func type: 2000 
# container_startup_time = 10 #sec
# container_showdown_time = 10 #sec
# container_idle_resource = 100 #100mi CPU
# container_startup_resource = 200 #100mi CPU
# number_of_node = 8
# epoch_length = 900
# real_time_node_load_arr = np.zeros((number_of_node, epoch_length))
# number_of_core_per_region = 128
# max_cpu_time = number_of_node*number_of_core_per_region*number_of_node_per_region
# this_interval_avail_resource_time = np.ones(epoch_length,dtype=float)*max_cpu_time
# next_interval_avail_resource_time = np.ones(epoch_length,dtype=float)*max_cpu_time
# resource_per_request = 2
# ddl_laxity = 10
# # resource = 0.1
# # per_node_cooling_efficiency = []
# global_min_weight = 0.0005
# first_interval_flag = 1

# # container_startup_time = 0 #sec
# # container_showdown_time = 0 #sec
# # container_idle_resource = 0 #100mi CPU
# # container_startup_resource = 0 #100mi CPU

# vm_startup_time = 0
# vm_shutdown_time = 30

# idle_res_usage = 2

# carbon_density_list = [241.7, 221.5, 210.5, 202.1, 199.4, 199.8, 203.9, 223.1, 
#                            232.0, 244.1, 229.6, 228.9, 229.9, 227.0, 229.9, 249.0,
#                            246.2, 259.6, 273.2, 280.9, 282.4, 279.3,274.4, 260.4] #gram/kWh

# water_density = 0.53 #L/KWh
# cop_factor_list = [0.85, 0.92, 1.02, 1.10, 1.14, 1.20, 1.21, 1.22, 
#                     1.21, 1.20, 1.15, 1.11, 1.03, 0.96, 0.91, 0.816, 
#                     0.83, 0.82, 0.82, 0.81, 0.81, 0.80, 0.80, 0.80]
# cop = 1.5
# price_list = [10.0, 10.0 ,10.0 ,10.0, 10.0, 10.0, 11.5, 11.5, 
#                 11.5, 11.5, 11.5, 11.5, 11.5, 18.7, 18.7, 18.7, 
#                 18.7, 18.7, 11.5, 11.5, 11.5, 11.5, 11.5, 10.0]
