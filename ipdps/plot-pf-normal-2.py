import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm 
import statistics 

values_arr =np.loadtxt('outputs/azure_hybrid_v5_n4_d1_r20000_e4_change', delimiter=',')
fig, axs = plt.subplots(1, 3, figsize=(6, 3))
fig.tight_layout(pad=2)

greenc_values_arr=np.array([4.344101875,34.25494381106193,0.31055455369744844,1.4211559395583235]) #GreenC
kimchi_values_arr=np.array([2.9131481249999998,34.223760568670066,0.3870588941587782,1.8895661616446384]) #Kimchi
gofs_values_arr=np.array([4.386000000000001,33.33199869502764,0.3272203417333239,1.434753979048178]) #GOFS (Baseline)

# greenc_values_arr = greenc_values_arr/hybrid_values_arr
# values_arr[:,0] = values_arr[:,0]/hybrid_values_arr[0]
# values_arr[:,1] = values_arr[:,1]/hybrid_values_arr[1]
# values_arr[:,2] = values_arr[:,2]/hybrid_values_arr[2]
# hybrid_values_arr = hybrid_values_arr/hybrid_values_arr

values_arr=values_arr/gofs_values_arr
greenc_values_arr=greenc_values_arr/gofs_values_arr
kimchi_values_arr=kimchi_values_arr/gofs_values_arr
gofs_values_arr=gofs_values_arr/gofs_values_arr
print(values_arr)
print(greenc_values_arr)
print(kimchi_values_arr)
print(gofs_values_arr)
# exit()

x = values_arr[:,0]
y = values_arr[:,1]


axs[0].scatter(x,y, label='SOFA', color='tab:green')
axs[0].scatter(greenc_values_arr[0], greenc_values_arr[1], label='GreenC', color='tab:blue')
axs[0].scatter(kimchi_values_arr[0], kimchi_values_arr[1], label='Kimchi', color='tab:orange')
axs[0].scatter(gofs_values_arr[0], gofs_values_arr[1], label='GOFS (Baseline)', color='tab:red', marker='o')
# axs[0].legend(loc='upper center',  bbox_to_anchor=(1.18, 1.18), ncol=4)
axs[0].set_xlabel("Normalized Utility")
axs[0].set_ylabel("Normalized Energy Cost")
# axs[0].set_yticks(np.arange(0.85, 1.25, 0.1))
axs[0].set_xticks(np.arange(0.6, 1.01, 0.1))

x = values_arr[:,2]
y = values_arr[:,1]
axs[1].scatter(x,y, label='SOFA', color='tab:green')
axs[1].scatter(greenc_values_arr[2], greenc_values_arr[1], label='GreenC', color='tab:blue')
axs[1].scatter(kimchi_values_arr[2], kimchi_values_arr[1], label='Kimchi', color='tab:orange')
axs[1].scatter(gofs_values_arr[2], gofs_values_arr[1], label='GOFS (Baseline)', color='tab:red', marker='o')

axs[1].legend(loc='upper center',  bbox_to_anchor=(0.5, 1.18), ncol=4)
axs[1].set_xlabel("Normalized Carbon")
axs[1].set_ylabel("Normalized Energy Cost")
# axs[1].set_yticks(np.arange(0.85, 1.25, 0.1))
# axs[1].set_yticks(np.arange(0.6, 1.5, 0.2))
axs[1].set_xticks(np.arange(0.8, 1.21, 0.1))

x = values_arr[:,3]
y = values_arr[:,1]
axs[2].scatter(x,y, label='SOFA', color='tab:green')
axs[2].scatter(greenc_values_arr[3], greenc_values_arr[1], label='GreenC', color='tab:blue')
axs[2].scatter(kimchi_values_arr[3], kimchi_values_arr[1], label='Kimchi', color='tab:orange')
axs[2].scatter(gofs_values_arr[3], gofs_values_arr[1], label='GOFS (Baseline)', color='tab:red', marker='o')
axs[2].set_xlabel("Normalized Water")
axs[2].set_ylabel("Normalized Energy Cost")
# axs[2].set_yticks(np.arange(0.85, 1.25, 0.1))
# axs[2].set_yticks(np.arange(0.6, 1.3, 0.1))
axs[2].set_xticks(np.arange(0.7, 1.51, 0.2))

# plt.savefig('plot-pf.jpeg', dpi=300, bbox_inches="tight")
plt.savefig('plot-pf-normal-2.jpeg', dpi=1200)