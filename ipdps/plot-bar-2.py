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
	A=np.array([2.142962590296702,27.69341868244589,0.2525678594063145,1.3762990751445592,
             2.9898406652869967,27.885469187238346,0.27711015708370607,1.4449213834410233,
             4.251584633794407,32.0732675642699,0.2940016275489995,1.55579824091427,
             5.361092717105629,35.09896599625586,0.30412266972068885,1.705437003931287])
	B=np.array([2.169435,28.232035784485674,0.26350706187356976,1.4128519685452692,
             3.2541525,31.724003074281843,0.28831515055998697,1.5672293566769528,
             4.33887,34.00993818839786,0.3066419706976644,1.673778476219294,
             5.4235875,35.866655966071555,0.32257097917895344,1.7630385774634034])
	C=np.array([2.122305,26.839839169873375,0.30354950394902575,1.4818835760298408,
             3.1834575,30.893847311259197,0.34941639552553944,1.7057581241550044,
             4.24461,34.223760568670066,0.3870588941587782,1.8895661616446384,
             5.3057625,36.84759062983488,0.4167430950736638,2.0343790868120433])
	D=np.array([2.1930000000000005,28.045121276889994,0.2751984156208763,1.438614826473212,
             3.2895,31.875006529824525,0.3075426661958702,1.6213933019452955,
             4.386000000000001,34.592871567019046,0.33346211213752447,1.7589326499837747,
             4.386000000000001,34.592871567019046,0.33346211213752447,1.7589326499837747])

	ref=np.array([2.1930000000000005,28.045121276889994,0.2751984156208763,1.438614826473212,
             2.1930000000000005,28.045121276889994,0.2751984156208763,1.438614826473212,
             2.1930000000000005,28.045121276889994,0.2751984156208763,1.438614826473212,
             2.1930000000000005,28.045121276889994,0.2751984156208763,1.438614826473212])
 
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

	axs[0, 0].bar(x - 1.5 * width, a, width=width, label='10000x Trace', color='tab:green')
	axs[0, 0].bar(x - 0.5 * width, b,  width=width, label='15000x Trace', color='tab:blue')
	axs[0, 0].bar(x + 0.5 * width, c, width=width, label='20000x Trace',color='tab:orange')
	axs[0, 0].bar(x + 1.5 * width, d, width=width, label='25000x Trace',color='tab:purple')
	# axs[0, 0].bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

	# axs[0, 0].text(x[0], -0.15*5, 'SOFA', fontsize = 7, horizontalalignment='center')
	# axs[0, 0].text(x[1], -0.15*5, 'GreenC', fontsize = 7, horizontalalignment='center')
	# axs[0, 0].text(x[2], -0.15*5, 'Kimchi', fontsize = 7, horizontalalignment='center')
	# axs[0, 0].text(x[3], -0.15*5, 'GOFS', fontsize = 7, horizontalalignment='center')

	# axs[0, 0].text((x[1]+x[2])/2, -0.15*10*4, 'Framework', fontsize = 10, horizontalalignment='center')
	ax=axs[0, 0]
	ax.set_xticks(np.arange(0-width, 3.5, 1))
	ax.set_yticks(np.arange(0, 3.1, 0.6))
	labels = [item.get_text() for item in ax.get_xticklabels()]
	labels = ['SOFA', 'GreenC', 'Kimchi', 'GOFS']
	ax.set_xticklabels(labels)

	plt.sca(axs[0, 0])
	# plt.xticks([])
	plt.yticks(fontsize=14)
	plt.xticks(fontsize=14)
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
	ax.set_yticks(np.arange(0, 1.6, 0.3))
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
	ax.set_yticks(np.arange(0, 1.6, 0.3))
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
	ax.set_yticks(np.arange(0, 1.6, 0.3))
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

	figure.savefig('plot-bar-2.jpeg', dpi=300, bbox_inches="tight")
	
	
if __name__ == '__main__':
	main()
