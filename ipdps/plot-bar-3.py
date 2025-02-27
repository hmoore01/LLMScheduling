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
	A=np.array([4.159305289827214,27.666029659051443,0.32793336518714883,1.3696854780730323,
             4.251584633794407,32.0732675642699,0.2940016275489995,1.55579824091427,
             4.134410770961981,26.59782436192689,0.32744317648526644,1.4031279933298257,
             4.29764438194193,30.791866016844132,0.31663327350647097,1.2906864387573076])
	B=np.array([4.386000000000001,34.223760568670066,0.3870588941587782,1.8895661616446384,
             4.33887,34.00993818839786,0.3066419706976644,1.673778476219294,
             4.33887,28.626692698139088,0.3869527436805158,1.7457814193008918,
             4.344101875,34.25494381106193,0.31055455369744844,1.4211559395583235])
	C=np.array([4.386000000000001,34.223760568670066,0.3870588941587782,1.8895661616446384,
             4.24461,34.223760568670066,0.3870588941587782,1.8895661616446384,
             4.24461,34.223760568670066,0.3870588941587782,1.8895661616446384,
             2.9131481249999998,34.223760568670066,0.3870588941587782,1.8895661616446384])
	D=np.array([4.386000000000001,34.223760568670066,0.3870588941587782,1.88956616164463844,
             4.386000000000001,34.592871567019046,0.33346211213752447,1.7589326499837747,
             4.386000000000001,29.93777436692163,0.39638350407769685,1.8041758560257042,
             4.386000000000001,33.33199869502764,0.3272203417333239,1.434753979048178])

	ref=np.array([4.386000000000001,34.223760568670066,0.3870588941587782,1.88956616164463844,
             4.386000000000001,34.223760568670066,0.3870588941587782,1.88956616164463844,
             4.386000000000001,34.223760568670066,0.3870588941587782,1.88956616164463844,
             4.386000000000001,34.223760568670066,0.3870588941587782,1.88956616164463844])
 
	A=A/ref
	B=B/ref
	C=C/ref
	D=D/ref
	## req test
	# A=np.array([1.85,0.05,3.50,0.09,2.34,0.06,4.42,0.16,2.96,0.09,5.62,0.26,4.03,0.12,7.70,0.41])
	# B=np.array([3.95,0.02,7.54,0.19,4.35,0.06,8.30,0.26,4.86,0.13,9.29,0.39,5.49,0.36,10.49,0.59])
	# C=np.array([5.15,0.08,9.80,0.35,5.37,0.12,10.23,0.41,5.65,0.220.78,0.52,5.99,0.46,11.43,0.67])
	# D=np.array([0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00])

	fig, axs = plt.subplots(4, 1)
	figure = plt.gcf() # get current figure
	figure.set_size_inches(10, 8)

	a=[0]*4
	b=[0]*4
	c=[0]*4
	d=[0]*4

	j=0
	the_list=[A,B,C,D]
	for i in range(4):
		a[i]=the_list[i][j]
	for i in range(4):
		b[i]=the_list[i][j+4]
	for i in range(4):
		c[i]=the_list[i][j+8]
	for i in range(4):
		d[i]=the_list[i][j+12]

	size = len(a)
	x = np.arange(size)

	total_width, n = 0.8, 4
	width = total_width / n
	x = x - (total_width - width) / 3

	axs[0].bar(x - 1.5 * width, a, width=width, label='Even Origin', color='tab:green')
	axs[0].bar(x - 0.5 * width, b,  width=width, label='Coastal Origin', color='tab:blue')
	axs[0].bar(x + 0.5 * width, c, width=width, label='Centre Origin',color='tab:orange')
	axs[0].bar(x + 1.5 * width, d, width=width, label='Flowing Origin',color='tab:purple')
	# axs[0].bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

	# axs[0].text(x[0], -0.15*5, 'SOFA', fontsize = 7, horizontalalignment='center')
	# axs[0].text(x[1], -0.15*5, 'GreenC', fontsize = 7, horizontalalignment='center')
	# axs[0].text(x[2], -0.15*5, 'Kimchi', fontsize = 7, horizontalalignment='center')
	# axs[0].text(x[3], -0.15*5, 'GOFS', fontsize = 7, horizontalalignment='center')

	# axs[0].text((x[1]+x[2])/2, -0.15*10*4, 'Framework', fontsize = 10, horizontalalignment='center')
	ax=axs[0]
	ax.set_xticks(np.arange(0-width, 3.5, 1))
	ax.set_yticks(np.arange(0, 1.3, 0.3))
	labels = [item.get_text() for item in ax.get_xticklabels()]
	labels = ['SOFA', 'GreenC', 'Kimchi', 'GOFS']
	ax.set_xticklabels(labels)

	plt.sca(axs[0])
	# plt.xticks([])
	plt.yticks(fontsize=14)
	plt.xticks(fontsize=13)
	plt.ylabel("Normalized\n Utility", fontsize=14)

	axs[0].grid(axis='y', linestyle='--', color='lightgrey')
	axs[0].legend(loc='center', bbox_to_anchor=(0.48, 1.22), fontsize=13, ncol=4)
	# axs[0].title("Normalized EDP Results")
	


	a=[0]*4
	b=[0]*4
	c=[0]*4
	d=[0]*4

	j=1
	the_list=[A,B,C,D]
	for i in range(4):
		a[i]=the_list[i][j]
	for i in range(4):
		b[i]=the_list[i][j+4]
	for i in range(4):
		c[i]=the_list[i][j+8]
	for i in range(4):
		d[i]=the_list[i][j+12]

	size = len(a)
	x = np.arange(size)

	total_width, n = 0.8, 4
	width = total_width / n
	x = x - (total_width - width) / 3

	# axs[1].axhline(y=10, color="red", label='Preset 10% SLO Violation Rate')
	# axs[1].legend(loc='upper right')

	axs[1].bar(x - 1.5 * width, a, width=width, label='2 Nodes', color='tab:green')
	axs[1].bar(x - 0.5 * width, b,  width=width, label='4 Nodes', color='tab:blue')
	axs[1].bar(x + 0.5 * width, c, width=width, label='8 Nodes',color='tab:orange')
	axs[1].bar(x + 1.5 * width, d, width=width, label='16 Nodes',color='tab:purple')
	
	# axs[0].bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
	

	# axs[1].text(x[0], -0.15*5*5, 'SOFA', fontsize = 7, horizontalalignment='center')
	# axs[1].text(x[1], -0.15*5*5, 'GreenC', fontsize = 7, horizontalalignment='center')
	# axs[1].text(x[2], -0.15*5*5, 'Kimchi', fontsize = 7, horizontalalignment='center')
	# axs[1].text(x[3], -0.15*5*5, 'GOFS', fontsize = 7, horizontalalignment='center')

	ax=axs[1]
	ax.set_xticks(np.arange(0-width, 3.5, 1))
	ax.set_yticks(np.arange(0, 1.3, 0.3))
	labels = [item.get_text() for item in ax.get_xticklabels()]
	labels = ['SOFA', 'GreenC', 'Kimchi', 'GOFS']
	ax.set_xticklabels(labels)
	# axs[1].text((x[1]+x[2])/2, -0.15*10*5, 'Framework', fontsize = 10, horizontalalignment='center')

	plt.sca(axs[1])

	plt.yticks(fontsize=14)
	plt.xticks(fontsize=13)
	plt.ylabel("Normalized\n Energy Cost", fontsize=14)
	# plt.ylim(0,90)
	

	axs[1].grid(axis='y', linestyle='--', color='lightgrey')




	j=2
	the_list=[A,B,C,D]
	for i in range(4):
		a[i]=the_list[i][j]
	for i in range(4):
		b[i]=the_list[i][j+4]
	for i in range(4):
		c[i]=the_list[i][j+8]
	for i in range(4):
		d[i]=the_list[i][j+12]

	size = len(a)
	x = np.arange(size)

	total_width, n = 0.8, 4
	width = total_width / n
	x = x - (total_width - width) / 3

	axs[2].bar(x - 1.5 * width, a, width=width, label='2 Nodes', color='tab:green')
	axs[2].bar(x - 0.5 * width, b,  width=width, label='4 Nodes', color='tab:blue')
	axs[2].bar(x + 0.5 * width, c, width=width, label='8 Nodes',color='tab:orange')
	axs[2].bar(x + 1.5 * width, d, width=width, label='16 Nodes',color='tab:purple')
	# axs[0].bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

	# axs[2].text(x[0], -0.15*5*2, 'SOFA', fontsize = 7, horizontalalignment='center')
	# axs[2].text(x[1], -0.15*5*2, 'GreenC', fontsize = 7, horizontalalignment='center')
	# axs[2].text(x[2], -0.15*5*2, 'Kimchi', fontsize = 7, horizontalalignment='center')
	# axs[2].text(x[3], -0.15*5*2, 'GOFS', fontsize = 7, horizontalalignment='center')
	ax=axs[2]
	ax.set_xticks(np.arange(0-width, 3.5, 1))
	ax.set_yticks(np.arange(0, 1.3, 0.3))
	labels = [item.get_text() for item in ax.get_xticklabels()]
	labels = ['SOFA', 'GreenC', 'Kimchi', 'GOFS']
	ax.set_xticklabels(labels)
	# axs[2].text((x[1]+x[2])/2, -0.15*10*2, 'Framework', fontsize = 10, horizontalalignment='center')

	plt.sca(axs[2])

	plt.yticks(fontsize=14)
	plt.xticks(fontsize=13)
	plt.ylabel("Normalized\n Carbon", fontsize=14)
	

	axs[2].grid(axis='y', linestyle='--', color='lightgrey')




	j=3
	the_list=[A,B,C,D]
	for i in range(4):
		a[i]=the_list[i][j]
	for i in range(4):
		b[i]=the_list[i][j+4]
	for i in range(4):
		c[i]=the_list[i][j+8]
	for i in range(4):
		d[i]=the_list[i][j+12]

	size = len(a)
	x = np.arange(size)

	total_width, n = 0.8, 4
	width = total_width / n
	x = x - (total_width - width) / 3

	axs[3].bar(x - 1.5 * width, a, width=width, label='2 Nodes', color='tab:green')
	axs[3].bar(x - 0.5 * width, b,  width=width, label='4 Nodes', color='tab:blue')
	axs[3].bar(x + 0.5 * width, c, width=width, label='8 Nodes',color='tab:orange')
	axs[3].bar(x + 1.5 * width, d, width=width, label='16 Nodes',color='tab:purple')
	# axs[0].bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

	# axs[2].text(x[0], -0.15*5*7, 'SOFA', fontsize = 7, horizontalalignment='center')
	# axs[2].text(x[1], -0.15*5*7, 'GreenC', fontsize = 7, horizontalalignment='center')
	# axs[2].text(x[2], -0.15*5*7, 'Kimchi', fontsize = 7, horizontalalignment='center')
	# axs[2].text(x[3], -0.15*5*7, 'GOFS', fontsize = 7, horizontalalignment='center')

	ax=axs[3]
	ax.set_xticks(np.arange(0-width, 3.5, 1))
	ax.set_yticks(np.arange(0, 1.3, 0.3))
	labels = [item.get_text() for item in ax.get_xticklabels()]
	labels = ['SOFA', 'GreenC', 'Kimchi', 'GOFS']
	ax.set_xticklabels(labels)
	# axs[2].text((x[1]+x[2])/2, -0.15*10*2, 'Framework', fontsize = 10, horizontalalignment='center')

	plt.sca(axs[3])

	plt.yticks(fontsize=14)
	plt.xticks(fontsize=13)
	plt.ylabel("Normalized\n Water", fontsize=14)
	# plt.ylim(0,90)
	

	axs[3].grid(axis='y', linestyle='--', color='lightgrey')

	figure.savefig('plot-bar-3.jpeg', dpi=300, bbox_inches="tight")
	
	
if __name__ == '__main__':
	main()
