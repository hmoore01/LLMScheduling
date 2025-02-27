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
	A=np.array([4.251584633794407,32.0732675642699,0.2940016275489995,1.55579824091427,
             4.0307050391831405,34.77187248015814,0.3840371858157947,1.8335308852523031,
             4.259745635042694,65.66377251861563,0.6347609691295877,3.192069258482484,
             3.6185930243588076,163.62091536679344,1.8527136037257754,9.02667095896649])
	B=np.array([4.33887,34.00993818839786,0.3066419706976644,1.673778476219294,
             4.344186103896104,44.4927249675477,0.42418244013424805,2.2542611143329143,
             4.28435,84.16338621589777,0.9027446374903437,4.524630255137337,
             2.625496015,303.0256548702326,3.065634515447246,15.777348405065384])
	C=np.array([4.24461,34.223760568670066,0.3870588941587782,1.8895661616446384,
             4.24461,41.71472121999232,0.47176542195503063,2.3030560474963413,
             4.24461,83.06804083222254,0.9386291786404114,4.590908348462532,
             2.6227248585000003,274.9254554261646,3.112435607807763,15.189881658272954])
	D=np.array([4.386000000000001,34.592871567019046,0.33346211213752447,1.7589326499837747,
             4.386000000000001,46.63974906781051,0.4366916028538401,2.336447031510213,
             2.6767193000000002,185.75614511189423,1.218770682589391,7.9564986898896795,
             0.7726017000000001,275.9484639342526,1.733574661353754,11.614338795266097])

	ref=np.array([4.24461,34.223760568670066,0.3870588941587782,1.8895661616446384,
             4.24461,34.223760568670066,0.3870588941587782,1.8895661616446384,
             4.24461,34.223760568670066,0.3870588941587782,1.8895661616446384,
             4.24461,34.223760568670066,0.3870588941587782,1.8895661616446384])
 
	A=A/ref
	B=B/ref
	C=C/ref
	D=D/ref
	## req test
	# A=np.array([1.85,0.05,3.50,0.09,2.34,0.06,4.42,0.16,2.96,0.09,5.62,0.26,4.03,0.12,7.70,0.41])
	# B=np.array([3.95,0.02,7.54,0.19,4.35,0.06,8.30,0.26,4.86,0.13,9.29,0.39,5.49,0.36,10.49,0.59])
	# C=np.array([5.15,0.08,9.80,0.35,5.37,0.12,10.23,0.41,5.65,0.21,10.78,0.52,5.99,0.46,11.43,0.67])
	# D=np.array([0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00])

	fig, axs = plt.subplots(2, 2)
	figure = plt.gcf() # get current figure
	figure.set_size_inches(10, 6)

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

	axs[0, 0].bar(x - 1.5 * width, a, width=width, label='1x Duration', color='tab:green')
	axs[0, 0].bar(x - 0.5 * width, b,  width=width, label='2x Duration', color='tab:blue')
	axs[0, 0].bar(x + 0.5 * width, c, width=width, label='4x Duration',color='tab:orange')
	axs[0, 0].bar(x + 1.5 * width, d, width=width, label='8x Duration',color='tab:purple')
	# axs[0, 0].bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

	# axs[0, 0].text(x[0], -0.15*5, 'SOFA', fontsize = 7, horizontalalignment='center')
	# axs[0, 0].text(x[1], -0.15*5, 'GreenC', fontsize = 7, horizontalalignment='center')
	# axs[0, 0].text(x[2], -0.15*5, 'Kimchi', fontsize = 7, horizontalalignment='center')
	# axs[0, 0].text(x[3], -0.15*5, 'GOFS', fontsize = 7, horizontalalignment='center')

	# axs[0, 0].text((x[1]+x[2])/2, -0.15*10*4, 'Framework', fontsize = 10, horizontalalignment='center')
	ax=axs[0, 0]
	ax.set_xticks(np.arange(0-width, 3.5, 1))
	# ax.set_yticks(np.arange(0, 3.1, 0.6))
	labels = [item.get_text() for item in ax.get_xticklabels()]
	labels = ['SOFA', 'GreenC', 'Kimchi', 'GOFS']
	ax.set_xticklabels(labels)

	plt.sca(axs[0, 0])
	# plt.xticks([])
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.ylabel("Normalized Utility", fontsize=14)

	axs[0, 0].grid(axis='y', linestyle='--', color='lightgrey')
	axs[0, 0].legend(loc='center', bbox_to_anchor=(1.05, 1.2), fontsize=13, ncol=4)
	# axs[0, 0].title("Normalized EDP Results")
	


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

	# axs[0, 1].axhline(y=10, color="red", label='Preset 10% SLO Violation Rate')
	# axs[0, 1].legend(loc='upper right')
	# axs[0, 1].set_yticks(np.arange(0, 100, 20))

	axs[0, 1].bar(x - 1.5 * width, a, width=width, label='2 Nodes', color='tab:green')
	axs[0, 1].bar(x - 0.5 * width, b,  width=width, label='4 Nodes', color='tab:blue')
	axs[0, 1].bar(x + 0.5 * width, c, width=width, label='8 Nodes',color='tab:orange')
	axs[0, 1].bar(x + 1.5 * width, d, width=width, label='16 Nodes',color='tab:purple')
	
	# axs[0, 0].bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
	

	# axs[0, 1].text(x[0], -0.15*5*5, 'SOFA', fontsize = 7, horizontalalignment='center')
	# axs[0, 1].text(x[1], -0.15*5*5, 'GreenC', fontsize = 7, horizontalalignment='center')
	# axs[0, 1].text(x[2], -0.15*5*5, 'Kimchi', fontsize = 7, horizontalalignment='center')
	# axs[0, 1].text(x[3], -0.15*5*5, 'GOFS', fontsize = 7, horizontalalignment='center')

	ax=axs[0, 1]
	ax.set_xticks(np.arange(0-width, 3.5, 1))
	# ax.set_yticks(np.arange(0, 1.6, 0.3))
	labels = [item.get_text() for item in ax.get_xticklabels()]
	labels = ['SOFA', 'GreenC', 'Kimchi', 'GOFS']
	ax.set_xticklabels(labels)
	# axs[0, 1].text((x[1]+x[2])/2, -0.15*10*5, 'Framework', fontsize = 10, horizontalalignment='center')

	plt.sca(axs[0, 1])

	plt.yticks(fontsize=14)
	plt.xticks(fontsize=14)
	plt.ylabel("Normalized Energy Cost", fontsize=14)
	# plt.ylim(0,90)
	

	axs[0, 1].grid(axis='y', linestyle='--', color='lightgrey')




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

	axs[1,0].bar(x - 1.5 * width, a, width=width, label='2 Nodes', color='tab:green')
	axs[1,0].bar(x - 0.5 * width, b,  width=width, label='4 Nodes', color='tab:blue')
	axs[1,0].bar(x + 0.5 * width, c, width=width, label='8 Nodes',color='tab:orange')
	axs[1,0].bar(x + 1.5 * width, d, width=width, label='16 Nodes',color='tab:purple')
	# axs[0, 0].bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

	# axs[1,0].text(x[0], -0.15*5*2, 'SOFA', fontsize = 7, horizontalalignment='center')
	# axs[1,0].text(x[1], -0.15*5*2, 'GreenC', fontsize = 7, horizontalalignment='center')
	# axs[1,0].text(x[2], -0.15*5*2, 'Kimchi', fontsize = 7, horizontalalignment='center')
	# axs[1,0].text(x[3], -0.15*5*2, 'GOFS', fontsize = 7, horizontalalignment='center')
	ax=axs[1, 0]
	ax.set_xticks(np.arange(0-width, 3.5, 1))
	# ax.set_yticks(np.arange(0, 1.6, 0.3))
	labels = [item.get_text() for item in ax.get_xticklabels()]
	labels = ['SOFA', 'GreenC', 'Kimchi', 'GOFS']
	ax.set_xticklabels(labels)
	# axs[1,0].text((x[1]+x[2])/2, -0.15*10*2, 'Framework', fontsize = 10, horizontalalignment='center')

	plt.sca(axs[1,0])

	plt.yticks(fontsize=14)
	plt.xticks(fontsize=14)
	plt.ylabel("Normalized Carbon", fontsize=14)
	

	axs[1,0].grid(axis='y', linestyle='--', color='lightgrey')




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

	axs[1,1].bar(x - 1.5 * width, a, width=width, label='2 Nodes', color='tab:green')
	axs[1,1].bar(x - 0.5 * width, b,  width=width, label='4 Nodes', color='tab:blue')
	axs[1,1].bar(x + 0.5 * width, c, width=width, label='8 Nodes',color='tab:orange')
	axs[1,1].bar(x + 1.5 * width, d, width=width, label='16 Nodes',color='tab:purple')
	# axs[0, 0].bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

	# axs[1,1].text(x[0], -0.15*5*7, 'SOFA', fontsize = 7, horizontalalignment='center')
	# axs[1,1].text(x[1], -0.15*5*7, 'GreenC', fontsize = 7, horizontalalignment='center')
	# axs[1,1].text(x[2], -0.15*5*7, 'Kimchi', fontsize = 7, horizontalalignment='center')
	# axs[1,1].text(x[3], -0.15*5*7, 'GOFS', fontsize = 7, horizontalalignment='center')

	ax=axs[1, 1]
	ax.set_xticks(np.arange(0-width, 3.5, 1))
	# ax.set_yticks(np.arange(0, 1.6, 0.3))
	labels = [item.get_text() for item in ax.get_xticklabels()]
	labels = ['SOFA', 'GreenC', 'Kimchi', 'GOFS']
	ax.set_xticklabels(labels)
	# axs[1,0].text((x[1]+x[2])/2, -0.15*10*2, 'Framework', fontsize = 10, horizontalalignment='center')

	plt.sca(axs[1,1])

	plt.yticks(fontsize=14)
	plt.xticks(fontsize=14)
	plt.ylabel("Normalized Water", fontsize=14)
	# axs[1, 1].set_yticks(np.arange(0, 100, 20))
	# plt.ylim(0,90)
	

	axs[1,1].grid(axis='y', linestyle='--', color='lightgrey')

	figure.savefig('plot-dur.jpeg', dpi=300, bbox_inches="tight")
	
	
if __name__ == '__main__':
	main()
