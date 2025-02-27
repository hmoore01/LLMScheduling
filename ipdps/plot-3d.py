from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

my_data = np.genfromtxt('temp.csv', delimiter=' ')
# my_data = np.genfromtxt('temp_nor_random_100e.csv', delimiter=' ')
# print(my_data)
utility = my_data[:,0]
carbon = my_data[:,2]
water = my_data[:,3]

ax.scatter3D(carbon, water, utility, color = "green", label='Hybrid Search')
ax.set_xlabel('carbon', fontweight ='bold') 
ax.set_ylabel('water', fontweight ='bold') 
ax.set_zlabel('utility', fontweight ='bold')

ax.scatter3D(0.08858416486391679, 0.6979794966099933, 1.231932008526233, color = "red", label='Greencourier')
ax.scatter3D(0.11230697362017303, 0.5480848803274520, 1.17288, color = "orange", label='Kimchi')
ax.scatter3D(0.09401887670367778, 0.4970059387957992, 1.17, color = "blue", label='GOFS')

ax.legend()

plt.show()