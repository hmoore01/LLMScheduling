import numpy as np
import hvwfg
from sklearn.preprocessing import MinMaxScaler

standard_objective=[1.17288, 9.928444210768594, 0.11230697362017303, 0.548084880327452]
min_objs=[element * 0.5 for element in standard_objective]
max_objs=[element * 1.5 for element in standard_objective]
scaler = MinMaxScaler((0, 100))
reference_point=[100]*4
print(scaler.fit([min_objs, max_objs]))
    
my_data = np.genfromtxt('temp.csv', delimiter=' ')
my_data = np.genfromtxt('temp_nor_random_100e.csv', delimiter=' ')
pareto_front = my_data
# print(my_data)
for j in range(20):
    pareto_front[j][0]=100-pareto_front[j][0]
phv_value = hvwfg.wfg(np.array(scaler.transform(pareto_front)).astype('double'), np.array(scaler.transform([reference_point])[0]).astype('double'))
print(phv_value)