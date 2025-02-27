import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import shutil
import matplotlib.ticker as mtick
# import plotinpy as pnp

def main():

	# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
	
	## cluster test
	A=np.array([0.66099223,0.18237893,0.6600389,0.459996,0.659707,0.265048,0.662445,0.610247,0.670465,0.329103,0.669626,0.675713,0.87719,0.490513,0.875713,0.821087,0.896459,1.062294,0.890385,0.816032])
	B=np.array([0.90663447,0.54941756,0.90905963,0.775247,0.915255,0.784114,0.917656,0.891202,0.972432,1.018366,0.973547,1.003406,1.022184,1.132346,1.022709,1.107122,1.061715,1.245195,1.061527,1.193453])
	C=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
	D=np.array([0.82783381,0.47764809,0.83150038,0.614214,0.897275,0.710217,0.899649,0.768742,0.957064,0.986837,0.95757,0.882545,0,0,0,0,0,0,0,0])

	## req test
	A=np.array([0.66099223,11.5772057,0.6600389,0.459996,0.659707,12.23653,0.662445,0.610247,0.670465,11.71401,0.669626,0.675713,0.87719,17.43836,0.875713,0.821087,0.920593162,23.62203649,0.915652413,0.874165164])
	B=np.array([0.90663447,34.8763969,0.90905963,0.775247,0.915255,36.20035,0.917656,0.891202,0.972432,36.24742,0.973547,1.003406,1.022184,40.25631,1.022709,1.107122,1.061715,40.27119,1.061527,1.193453])
	C=np.array([1,63.4788541,1,1,1,46.16722,1,1,1,35.59372,1,1,1,35.55124,1,1,1,32.34127,1,1])
	D=np.array([0.82783381,30.3205532,0.83150038,0.614214,
			    0.897275,32.78873,0.899649,0.768742,
				0.957064,35.12519,0.95757,0.882545,
				0,0,0,0,0,0,0,0])

	fig, axs = plt.subplots(2, 2)
	figure = plt.gcf() # get current figure
	figure.set_size_inches(10, 6)

	a=[0]*4
	b=[0]*4
	c=[0]*4
	d=[0]*4
	e=[0]*4

	j=0
	the_list=[C,B,D,A]
	for i in range(4):
		a[i]=the_list[i][j]
	for i in range(4):
		b[i]=the_list[i][j+4]
	for i in range(4):
		c[i]=the_list[i][j+8]
	for i in range(4):
		d[i]=the_list[i][j+12]
	for i in range(4):
		e[i]=the_list[i][j+16]

	size = len(a)
	x = np.arange(size)

	total_width, n = 0.8, 5
	width = total_width / n
	x = x - (total_width - width) / 4

	axs[0, 0].bar(x - 2.0 * width, a, width=width, label='2 Nodes with 20X Trace', color='tab:green')
	axs[0, 0].bar(x - 1.0 * width, b,  width=width, label='4 Nodes with 40X Trace', color='tab:blue')
	axs[0, 0].bar(x + 0.0 * width, c, width=width, label='8 Nodes with 80X Trace',color='tab:orange')
	axs[0, 0].bar(x + 1.0 * width, d, width=width, label='16 Nodes with 160X Trace',color='tab:purple')
	axs[0, 0].bar(x + 2.0 * width, e, width=width, label='32 Nodes with 320X Trace',color='tab:red')

	axs[0, 0].set_xticks([0-1/6,1-1/6,2-1/6,3-1/6])
	labels = [item.get_text() for item in axs[0, 0].get_xticklabels()]
	labels = ['Hybrid','Score','RL','CASA']
	axs[0, 0].set_xticklabels(labels, fontsize = 10)
	axs[0, 0].tick_params('x', length=0, width=2, which='major')
	axs[0, 0].set_yticks(np.arange(0.2, 1.5, 0.2))

	# axs[0, 0].text((x[1]+x[2])/2, -0.15*10*4, 'Framework', fontsize = 10, horizontalalignment='center')

	plt.sca(axs[0, 0])
	# plt.xticks([])
	plt.yticks(fontsize=14)
	plt.ylabel("Normalized Dollar Cost", fontsize=10)

	axs[0, 0].grid(axis='y', linestyle='--', color='lightgrey')
	axs[0, 0].legend(loc='center', bbox_to_anchor=(1.1, 1.2), fontsize=10, ncol=3)
	# axs[0, 0].title("Normalized EDP Results")
	


	a=[0]*4
	b=[0]*4
	c=[0]*4
	d=[0]*4

	j=1
	the_list=[C,B,D,A]
	for i in range(4):
		a[i]=the_list[i][j]
	for i in range(4):
		b[i]=the_list[i][j+4]
	for i in range(4):
		c[i]=the_list[i][j+8]
	for i in range(4):
		d[i]=the_list[i][j+12]
	for i in range(4):
		e[i]=the_list[i][j+16]

	size = len(a)
	x = np.arange(size)

	total_width, n = 0.8, 5
	width = total_width / n
	x = x - (total_width - width) / 4

	# axs[0, 1].axhline(y=10, color="red", label='Preset 10% SLO Violation Rate')
	# axs[0, 1].legend(loc='upper right')
	# axs[0, 1].set_yticks(np.arange(0, 100, 20))

	axs[0, 1].bar(x - 2.0 * width, a, width=width, label='2 Nodes', color='tab:green')
	axs[0, 1].bar(x - 1.0 * width, b,  width=width, label='4 Nodes', color='tab:blue')
	axs[0, 1].bar(x + 0.0 * width, c, width=width, label='8 Nodes',color='tab:orange')
	axs[0, 1].bar(x + 1.0 * width, d, width=width, label='16 Nodes',color='tab:purple')
	axs[0, 1].bar(x + 2.0 * width, e, width=width, label='32 Nodes',color='tab:red')
	
	# axs[0, 0].bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
	

	# axs[0, 1].text(x[0], -0.15*5*5, 'Dual Optimizer', fontsize = 7, horizontalalignment='center')
	# axs[0, 1].text(x[1], -0.15*5*5, 'Score', fontsize = 7, horizontalalignment='center')
	# axs[0, 1].text(x[2], -0.15*5*5, 'Hybrid', fontsize = 7, horizontalalignment='center')
	# axs[0, 1].text(x[3], -0.15*5*5, 'RL', fontsize = 7, horizontalalignment='center')

	axs[0, 1].set_xticks([0-1/6,1-1/6,2-1/6,3-1/6])
	labels = [item.get_text() for item in axs[0, 0].get_xticklabels()]
	labels = ['Hybrid','Score','RL','CASA']
	axs[0, 1].set_xticklabels(labels, fontsize = 10)
	axs[0, 1].tick_params('x', length=0, width=2, which='major')
	# axs[0, 1].set_yticks(np.arange(0.2, 1.5, 0.2))

	# axs[0, 1].text((x[1]+x[2])/2, -0.15*10*5, 'Framework', fontsize = 10, horizontalalignment='center')

	plt.sca(axs[0, 1])
	plt.yticks(fontsize=14)
	plt.ylabel("SLO Violation Rate (%)", fontsize=10)
	# plt.ylim(0,90)
	

	axs[0, 1].grid(axis='y', linestyle='--', color='lightgrey')




	j=2
	the_list=[C,B,D,A]
	for i in range(4):
		a[i]=the_list[i][j]
	for i in range(4):
		b[i]=the_list[i][j+4]
	for i in range(4):
		c[i]=the_list[i][j+8]
	for i in range(4):
		d[i]=the_list[i][j+12]
	for i in range(4):
		e[i]=the_list[i][j+16]

	size = len(a)
	x = np.arange(size)

	total_width, n = 0.8, 5
	width = total_width / n
	x = x - (total_width - width) / 4

	axs[1, 0].bar(x - 2.0 * width, a, width=width, label='2 Nodes', color='tab:green')
	axs[1, 0].bar(x - 1.0 * width, b,  width=width, label='4 Nodes', color='tab:blue')
	axs[1, 0].bar(x + 0.0 * width, c, width=width, label='8 Nodes',color='tab:orange')
	axs[1, 0].bar(x + 1.0 * width, d, width=width, label='16 Nodes',color='tab:purple')
	axs[1, 0].bar(x + 2.0 * width, e, width=width, label='32 Nodes',color='tab:red')
	# axs[0, 0].bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

	axs[1, 0].set_xticks([0-1/6,1-1/6,2-1/6,3-1/6])
	labels = [item.get_text() for item in axs[1, 0].get_xticklabels()]
	labels = ['Hybrid','Score','RL','CASA']
	axs[1, 0].set_xticklabels(labels, fontsize = 10)
	axs[1, 0].tick_params('x', length=0, width=2, which='major')
	axs[1, 0].set_yticks(np.arange(0.2, 1.5, 0.2))

	# axs[1,0].text((x[1]+x[2])/2, -0.15*10*2, 'Framework', fontsize = 10, horizontalalignment='center')

	plt.sca(axs[1,0])
	# plt.xticks([])
	plt.yticks(fontsize=14)
	plt.ylabel("Normalized Carbon", fontsize=10)
	

	axs[1,0].grid(axis='y', linestyle='--', color='lightgrey')




	j=3
	the_list=[C,B,D,A]
	for i in range(4):
		a[i]=the_list[i][j]
	for i in range(4):
		b[i]=the_list[i][j+4]
	for i in range(4):
		c[i]=the_list[i][j+8]
	for i in range(4):
		d[i]=the_list[i][j+12]
	for i in range(4):
		e[i]=the_list[i][j+16]

	size = len(a)
	x = np.arange(size)

	total_width, n = 0.8, 5
	width = total_width / n
	x = x - (total_width - width) / 4

	axs[1, 1].bar(x - 2.0 * width, a, width=width, label='2 Nodes', color='tab:green')
	axs[1, 1].bar(x - 1.0 * width, b,  width=width, label='4 Nodes', color='tab:blue')
	axs[1, 1].bar(x + 0.0 * width, c, width=width, label='8 Nodes',color='tab:orange')
	axs[1, 1].bar(x + 1.0 * width, d, width=width, label='16 Nodes',color='tab:purple')
	axs[1, 1].bar(x + 2.0 * width, e, width=width, label='32 Nodes',color='tab:red')

	axs[1, 1].set_xticks([0-1/6,1-1/6,2-1/6,3-1/6])
	labels = [item.get_text() for item in axs[1, 1].get_xticklabels()]
	labels = ['Hybrid','Score','RL','CASA']
	axs[1, 1].set_xticklabels(labels, fontsize = 10)
	axs[1, 1].tick_params('x', length=0, width=2, which='major')
	axs[1, 1].set_yticks(np.arange(0.2, 1.5, 0.2))

	# axs[1,0].text((x[1]+x[2])/2, -0.15*10*2, 'Framework', fontsize = 10, horizontalalignment='center')

	plt.sca(axs[1,1])
	plt.yticks(fontsize=14)
	plt.ylabel("Normalized System Load", fontsize=10)
	# axs[1, 1].set_yticks(np.arange(0, 100, 20))
	

	axs[1,1].grid(axis='y', linestyle='--', color='lightgrey')

	figure.savefig('plot-bar-4.jpeg', dpi=1200, bbox_inches="tight")
	
	
if __name__ == '__main__':
	main()
