import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import shutil
import matplotlib.ticker as mtick
# import plotinpy as pnp

def main():

	## req test
	A=np.array([4.3148767749241355,31.477772923005002,0.3077508361706136,1.2869718951539633]) #SOFA (utility, total_energy_cost, carbon, water)
	B=np.array([4.344101875,34.25494381106193,0.31055455369744844,1.4211559395583235]) #GreenC
	C=np.array([2.9131481249999998,34.223760568670066,0.3870588941587782,1.8895661616446384]) #Kimchi
	D=np.array([4.386000000000001,33.33199869502764,0.3272203417333239,1.434753979048178]) #GOFS

	A = A/D
	B = B/D
	C = C/D
	D = D/D
	# exit()
	print(A)
	print(B)
	print(C)
	print(D)

	fig, axs = plt.subplots(2, 2)
	figure = plt.gcf() # get current figure
	figure.set_size_inches(10, 6)

	a=[0]*4
	j=0
	the_list=[A,B,C,D]
	for i in range(4):
		a[i]=the_list[i][j]

	size = len(a)
	x = np.arange(size)

	ax =axs[0,0]
	print(a)
	# exit()
	ax.bar(x[0], a[0], label='SOFA', color='tab:green')
	ax.bar(x[1], a[1], label='GreenC', color='tab:blue')
	ax.bar(x[2], a[2], label='Kimchi', color='tab:orange')
	ax.bar(x[3], a[3], label='GOFS (Baseline)', color='tab:red')

	ax.set_yticks(np.arange(0, 1.1, 0.2))
	ax.set_xticks(np.arange(0, 4, 1))
	labels = [item.get_text() for item in ax.get_xticklabels()]
	labels = ['SOFA', 'GreenC', 'Kimchi', 'GOFS']
	ax.set_xticklabels(labels)
	ax.tick_params('x', length=0, width=2, which='major')
	plt.sca(ax)
	plt.ylim(0,1.1)
	# plt.xticks(np.arange(0, 1, width))
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.ylabel("Normalized Utility", fontsize=14)

	ax.grid(axis='y', linestyle='--', color='lightgrey')
	ax.legend(loc='center', bbox_to_anchor=(1.1, 1.2), fontsize=14, ncol=4)


	a=[0]*4
	j=1
	the_list=[A,B,C,D]
	for i in range(4):
		a[i]=the_list[i][j]
	ax =axs[0,1]

	ax.bar(x[0], a[0], label='SOFA', color='tab:green')
	ax.bar(x[1], a[1], label='GreenC', color='tab:blue')
	ax.bar(x[2], a[2], label='Kimchi', color='tab:orange')
	ax.bar(x[3], a[3], label='GOFS', color='tab:red')

	ax.set_yticks(np.arange(0, 1.1, 0.2))
	ax.set_xticks(np.arange(0, 4, 1))
	labels = [item.get_text() for item in ax.get_xticklabels()]
	labels = ['SOFA', 'GreenC', 'Kimchi', 'GOFS']
	ax.set_xticklabels(labels)
	ax.tick_params('x', length=0, width=2, which='major')
	plt.sca(ax)
	plt.ylim(0,1.1)
	# plt.xticks(np.arange(0, 1, width))
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.ylabel("Normalized Energy Cost", fontsize=14)

	ax.grid(axis='y', linestyle='--', color='lightgrey')

	a=[0]*4
	j=2
	the_list=[A,B,C,D]
	for i in range(4):
		a[i]=the_list[i][j]
	ax =axs[1,0]

	ax.bar(x[0], a[0], label='SOFA', color='tab:green')
	ax.bar(x[1], a[1], label='GreenC', color='tab:blue')
	ax.bar(x[2], a[2], label='Kimchi', color='tab:orange')
	ax.bar(x[3], a[3], label='GOFS', color='tab:red')

	ax.set_yticks(np.arange(0, 1.21, 0.2))
	# plt.ylim(0,1.2)
	ax.set_xticks(np.arange(0, 4, 1))
	labels = [item.get_text() for item in ax.get_xticklabels()]
	labels = ['SOFA', 'GreenC', 'Kimchi', 'GOFS']
	ax.set_xticklabels(labels)
	ax.tick_params('x', length=0, width=2, which='major')
	plt.sca(ax)
	plt.ylim(0,1.2)
	# plt.xticks(np.arange(0, 1, width))
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.ylabel("Normalized Carbon Emission", fontsize=14)

	ax.grid(axis='y', linestyle='--', color='lightgrey')

	a=[0]*4
	j=3
	the_list=[A,B,C,D]
	for i in range(4):
		a[i]=the_list[i][j]
	ax =axs[1,1]

	ax.bar(x[0], a[0], label='SOFA', color='tab:green')
	ax.bar(x[1], a[1], label='GreenC', color='tab:blue')
	ax.bar(x[2], a[2], label='Kimchi', color='tab:orange')
	ax.bar(x[3], a[3], label='GOFS', color='tab:red')

	ax.set_yticks(np.arange(0, 1.51, 0.3))
	# plt.ylim(0,1.5)
	ax.set_xticks(np.arange(0, 4, 1))
	labels = [item.get_text() for item in ax.get_xticklabels()]
	labels = ['SOFA', 'GreenC', 'Kimchi', 'GOFS']
	ax.set_xticklabels(labels)
	ax.tick_params('x', length=0, width=2, which='major')
	plt.sca(ax)
	# plt.xticks(np.arange(0, 1, width))
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.ylabel("Normalized Water Usage", fontsize=14)

	ax.grid(axis='y', linestyle='--', color='lightgrey')


	# axs[1,1].grid(axis='y', linestyle='--', color='lightgrey')
	figure.savefig('plot-1-azure.jpeg', dpi=1200, bbox_inches="tight")
	
	
if __name__ == '__main__':
	main()
