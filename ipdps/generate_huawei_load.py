import numpy as np
import pandas as pd
import pickle

dataframe=pd.read_csv('huawei_request.csv', index_col=None)
arr = dataframe.to_numpy()[:,2:]
arr = np.nan_to_num(arr)
arr = np.int_(arr)
# print(arr)
print(arr.shape)

func_distribution_arr=np.zeros((200,901),dtype=int)
func_distribution_arr_list=[func_distribution_arr]*96

for second in range(86400):
    epoch, second_in_epoch = divmod(second, 900)
    for func_id in range(200):
        if arr[second,func_id]:
            # print(second, func_id, arr[second,func_id])
            # func_distribution_arr_list[epoch][func_id,1+second_in_epoch]+=arr[second,func_id]
            # func_distribution_arr_list[epoch][func_id,0]+=arr[second,func_id]
            func_distribution_arr_list[epoch][func_id,1+second_in_epoch]+=int(arr[second,func_id]/200)
            func_distribution_arr_list[epoch][func_id,0]+=int(arr[second,func_id]/200)

# print(func_distribution_arr_list[0][4,:])
with open('huawei_func_distribution_list_original','wb') as f:
    pickle.dump(func_distribution_arr_list, f)
    
for epoch in range(96):
    request_intensity=0
    for func_id in range(200):
        request_intensity+=int(func_distribution_arr_list[epoch][func_id,0])
#     print(epoch, request_intensity)
# exit()
lut_arr=np.zeros((200,),dtype=int)
dataframe=pd.read_csv('huawei_execution.csv', index_col=None)
arr = dataframe.to_numpy()[:,2:]
arr = np.nan_to_num(arr)
# print(arr)
for func_id in range(200):
    count=0
    time=0
    for second in range(1440):
        if arr[second,func_id] > 0:
            count+=1
            time+=arr[second,func_id]
    if count:
        lut_arr[func_id]=int(time/count)
    else:
        lut_arr[func_id]=0
# print(lut_arr)

print(np.min(lut_arr[np.nonzero(lut_arr)]), np.max(lut_arr), np.average(lut_arr))
with open('huawei_func_runtime_list_original','wb') as f:
    pickle.dump(lut_arr, f)

    