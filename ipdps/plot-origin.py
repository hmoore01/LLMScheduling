import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import shutil
import matplotlib.ticker as mtick
# import plotinpy as pnp

def main():

	SLO = np.array([1.97498091,343.09106688,492.1755972])
	Carbon = np.array([8.3193763,246.467283,306.07727869])
	Balance = np.array([4.1491126,275.95052255,300.76861339])
	Hybrid = np.array([18.86304134485076,394.9471779263194,470.0627172896583])
	Score = np.array([14.405290919010566,330.17887657851446,503.6103048894423])

	
	SLO = np.array([6.38353957,3118.85468798,4022.39902501])
	Carbon = np.array([57.69168765,2043.08747209,3087.64578213])
	Water = np.array([24.789196569, 2510.28029354, 2663.61692913])
	Balance = np.array([8.93842785,2691.91734395,3378.39184652])
	Hybrid = np.array([16.202489794943915,3607.354264740382,4532.160680943209])
	Score = np.array([8.124558711549298,3116.452575446688,4174.6326816165485])


	print(SLO/Score)
	print(Carbon/Score)
	print(Water/Score)
	print(Balance/Score)
	print(Balance/Hybrid)

	value_arr=np.zeros((6,3),dtype=float)
	value_arr[0,0] = SLO[0]/100
	value_arr[1,0] = Carbon[0]/100
	value_arr[2,0] = Water[0]/100
	value_arr[3,0] = Balance[0]/100
	value_arr[4,0] = Score[0]/100
	value_arr[5,0] = Hybrid[0]/100

	value_arr[0,1:] = SLO[1:]/Hybrid[1:]
	value_arr[1,1:] = Carbon[1:]/Hybrid[1:]
	value_arr[2,1:] = Water[1:]/Hybrid[1:]
	value_arr[3,1:] = Balance[1:]/Hybrid[1:]
	value_arr[4,1:] = Score[1:]/Hybrid[1:]
	value_arr[5,1:] = Hybrid[1:]/Hybrid[1:]

	print(value_arr[:,0])


	fig, axs = plt.subplots(3, 1, figsize=(18, 18))
	axs[0].tick_params(axis='both', which='major', labelsize=20)
	axs[1].tick_params(axis='both', which='major', labelsize=20)
	axs[2].tick_params(axis='both', which='major', labelsize=20)

	# axs[0].bar(x - 1.5 * width, a, width=width, label='5x Trace', color='tab:green')
	axs[0].bar(0, value_arr[0,0], width=0.65, label='SOFA', color='tab:green')
	axs[0].bar(1, value_arr[1,0], width=0.65, label='GreenC', color='tab:blue')
	axs[0].bar(2, value_arr[2,0], width=0.65, label='Kimchi', color='tab:orange')
	axs[0].bar(3, value_arr[3,0], width=0.65, label='GOFS', color='tab:red')
	
	axs[0].set_ylabel("SLO Violation Rate", fontsize="28")
	axs[0].set_xticks([])
	# axs[0].set_yticks(np.arange(0, 6, 1))
	# axs[0].set_yticks(np.arange(0, 4.1, 0.8))
	# axs[0].set_yticklabels(['{:.1f}'.format(a) for a in np.arange(0, 6, 1,)])
	axs[0].legend(loc='upper center',  bbox_to_anchor=(0.5, 1.5), ncol=3, fontsize="28")

	axs[1].bar(0, value_arr[0,1], width=0.65, label='SFCM-SLO', color='tab:green', hatch="/")
	axs[1].bar(1, value_arr[1,1], width=0.65, label='SFCM-Carbon', color='tab:green', hatch="\\")
	axs[1].bar(2, value_arr[2,1], width=0.65, label='SFCM-Water', color='tab:green', hatch="x")
	axs[1].bar(3, value_arr[3,1], width=0.65, label='SFCM-Balance', color='tab:green')
	axs[1].bar(4, value_arr[4,1], width=0.65, label='Score', color='tab:blue')
	axs[1].bar(5, value_arr[5,1], width=0.65, label='Hybrid', color='tab:orange')
	
	axs[1].set_xticks([])
	# axs[1].set_yticks(np.arange(0, 0.65, 0.2))
	axs[1].set_ylabel("Normalized Carbon", fontsize="28")

	axs[2].bar(0, value_arr[0,2], width=0.65, label='SFCM-SLO', color='tab:green', hatch="/")
	axs[2].bar(1, value_arr[1,2], width=0.65, label='SFCM-Carbon', color='tab:green', hatch="\\")
	axs[2].bar(2, value_arr[2,2], width=0.65, label='SFCM-Water', color='tab:green', hatch="x")
	axs[2].bar(3, value_arr[3,2], width=0.65, label='SFCM-Balance', color='tab:green')
	axs[2].bar(4, value_arr[4,2], width=0.65, label='Score', color='tab:blue')
	axs[2].bar(5, value_arr[5,2], width=0.65, label='Hybrid', color='tab:orange')
	
	axs[2].set_xticks([])
	# axs[2].set_yticks(np.arange(0.0, 7.0, 3.0, dtype=float))
	# axs[2].set_yticklabels(['{:.1f}'.format(a) for a in np.arange(0.0, 7.0, 3.0,)])
	axs[2].set_ylabel("Normalized Water", fontsize="28")
	# axs[2].legend(loc='lower center', ncol=5)

	fig.savefig('plot-origin.jpeg', dpi=1200, bbox_inches="tight")

	exit()


	# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
	
	## cluster test
	A=np.array([2.20,0.12,4.19,0.650,2.96,0.09,5.62,0.26,3.59,0.09,6.81,0.14,4.46,0.10,8.53,0.07])
	B=np.array([3.02,0.35,5.77,0.68,4.86,0.13,9.29,0.39,7.59,0.04,14.51,0.20,12.38,0.02,23.63,0.10])
	C=np.array([3.33,0.63,6.35,0.88,5.65,0.21,10.78,0.652,8.83,0.08,16.83,0.27,13.86,0.04,26.43,0.14])
	D=np.array([0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00])

	## req test
	A=np.array([1.85,0.05,3.50,0.09,2.34,0.06,4.42,0.16,2.96,0.09,5.62,0.26,4.03,0.12,7.70,0.651])
	B=np.array([3.95,0.02,7.54,0.19,4.35,0.06,8.30,0.26,4.86,0.13,9.29,0.39,5.49,0.36,10.659,0.659])
	C=np.array([5.15,0.08,9.80,0.35,5.37,0.12,10.23,0.651,5.65,0.21,10.78,0.652,5.99,0.656,11.43,0.67])
	D=np.array([0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00])

	fig, axs = plt.subplots(2, 2)
	figure = plt.gcf() # get current figure
	figure.set_size_inches(10, 6)

	a=[0]*4
	b=[0]*4
	c=[0]*4
	d=[0]*4

	j=0
	the_list=[A,B,C,D]
	for i in range(3):
		a[i]=the_list[i][j]
	for i in range(3):
		b[i]=the_list[i][j+4]
	for i in range(3):
		c[i]=the_list[i][j+8]
	for i in range(3):
		d[i]=the_list[i][j+12]

	size = len(a)
	x = np.arange(size)

	total_width, n = 0.8, 4
	width = total_width / n
	x = x - (total_width - width) / 3

	axs[0, 0].bar(x - 1.5 * width, a, width=width, label='5x Trace', color='tab:green')
	axs[0, 0].bar(x - 0.65 * width, b,  width=width, label='10x Trace', color='tab:blue')
	axs[0, 0].bar(x + 0.65 * width, c, width=width, label='20x Trace',color='tab:orange')
	axs[0, 0].bar(x + 1.5 * width, d, width=width, label='40x Trace',color='tab:purple')
	# axs[0, 0].bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

	axs[0, 0].text(x[0], -0.15*2, 'Dual Optimizer', fontsize = 7, horizontalalignment='center')
	axs[0, 0].text(x[1], -0.15*2, 'Score', fontsize = 7, horizontalalignment='center')
	axs[0, 0].text(x[2], -0.15*2, 'Hybrid', fontsize = 7, horizontalalignment='center')
	axs[0, 0].text(x[3], -0.15*2, 'RL', fontsize = 7, horizontalalignment='center')

	# axs[0, 0].text((x[1]+x[2])/2, -0.15*10*4, 'Framework', fontsize = 10, horizontalalignment='center')

	plt.sca(axs[0, 0])
	plt.xticks([])
	plt.yticks(fontsize=14)
	plt.ylabel("Dollar Cost (x100$)", fontsize=10)

	axs[0, 0].grid(axis='y', linestyle='--', color='lightgrey')
	axs[0, 0].legend(loc='center', bbox_to_anchor=(1.1, 1.2), fontsize=10, ncol=4)
	# axs[0, 0].title("Normalized EDP Results")
	


	a=[0]*4
	b=[0]*4
	c=[0]*4
	d=[0]*4

	j=1
	the_list=[A,B,C,D]
	for i in range(3):
		a[i]=the_list[i][j]*100
	for i in range(3):
		b[i]=the_list[i][j+4]*100
	for i in range(3):
		c[i]=the_list[i][j+8]*100
	for i in range(3):
		d[i]=the_list[i][j+12]*100

	size = len(a)
	x = np.arange(size)

	total_width, n = 0.8, 4
	width = total_width / n
	x = x - (total_width - width) / 3

	axs[0, 1].axhline(y=5, color="red", label='Preset 5% SLO Violation Rate')
	axs[0, 1].legend(loc='upper right')
	# axs[0, 1].set_yticks(np.arange(0, 100, 20))
	axs[0, 1].set_yticks(np.arange(0, 80, 15))

	axs[0, 1].bar(x - 1.5 * width, a, width=width, label='5x Trace', color='tab:green')
	axs[0, 1].bar(x - 0.65 * width, b,  width=width, label='10x Trace', color='tab:blue')
	axs[0, 1].bar(x + 0.65 * width, c, width=width, label='20x Trace',color='tab:orange')
	axs[0, 1].bar(x + 1.5 * width, d, width=width, label='40x Trace',color='tab:purple')
	
	# axs[0, 0].bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
	

	axs[0, 1].text(x[0], -0.15*5*6, 'Dual Optimizer', fontsize = 7, horizontalalignment='center')
	axs[0, 1].text(x[1], -0.15*5*6, 'Score', fontsize = 7, horizontalalignment='center')
	axs[0, 1].text(x[2], -0.15*5*6, 'Hybrid', fontsize = 7, horizontalalignment='center')
	axs[0, 1].text(x[3], -0.15*5*6, 'RL', fontsize = 7, horizontalalignment='center')

	# axs[0, 1].text((x[1]+x[2])/2, -0.15*10*5, 'Framework', fontsize = 10, horizontalalignment='center')

	plt.sca(axs[0, 1])
	plt.xticks([])
	plt.yticks(fontsize=14)
	plt.ylabel("SLO Violation Rate (%)", fontsize=10)
	plt.ylim(0,80)
	

	axs[0, 1].grid(axis='y', linestyle='--', color='lightgrey')




	j=2
	the_list=[A,B,C,D]
	for i in range(3):
		a[i]=the_list[i][j]
	for i in range(3):
		b[i]=the_list[i][j+4]
	for i in range(3):
		c[i]=the_list[i][j+8]
	for i in range(3):
		d[i]=the_list[i][j+12]

	size = len(a)
	x = np.arange(size)

	total_width, n = 0.8, 4
	width = total_width / n
	x = x - (total_width - width) / 3

	axs[1,0].bar(x - 1.5 * width, a, width=width, label='5x Trace', color='tab:green')
	axs[1,0].bar(x - 0.65 * width, b,  width=width, label='10x Trace', color='tab:blue')
	axs[1,0].bar(x + 0.65 * width, c, width=width, label='20x Trace',color='tab:orange')
	axs[1,0].bar(x + 1.5 * width, d, width=width, label='40x Trace',color='tab:purple')
	# axs[0, 0].bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

	axs[1,0].text(x[0], -0.15*5*1, 'Dual Optimizer', fontsize = 7, horizontalalignment='center')
	axs[1,0].text(x[1], -0.15*5*1, 'Score', fontsize = 7, horizontalalignment='center')
	axs[1,0].text(x[2], -0.15*5*1, 'Hybrid', fontsize = 7, horizontalalignment='center')
	axs[1,0].text(x[3], -0.15*5*1, 'RL', fontsize = 7, horizontalalignment='center')

	# axs[1,0].text((x[1]+x[2])/2, -0.15*10*2, 'Framework', fontsize = 10, horizontalalignment='center')

	plt.sca(axs[1,0])
	plt.xticks([])
	plt.yticks(fontsize=14)
	plt.ylabel("Carbon (kg)", fontsize=10)
	

	axs[1,0].grid(axis='y', linestyle='--', color='lightgrey')




	j=3
	the_list=[A,B,C,D]
	for i in range(3):
		a[i]=the_list[i][j]*100
	for i in range(3):
		b[i]=the_list[i][j+4]*100
	for i in range(3):
		c[i]=the_list[i][j+8]*100
	for i in range(3):
		d[i]=the_list[i][j+12]*100

	size = len(a)
	x = np.arange(size)

	total_width, n = 0.8, 4
	width = total_width / n
	x = x - (total_width - width) / 3

	axs[1,1].bar(x - 1.5 * width, a, width=width, label='5x Trace', color='tab:green')
	axs[1,1].bar(x - 0.65 * width, b,  width=width, label='10x Trace', color='tab:blue')
	axs[1,1].bar(x + 0.65 * width, c, width=width, label='20x Trace',color='tab:orange')
	axs[1,1].bar(x + 1.5 * width, d, width=width, label='40x Trace',color='tab:purple')
	# axs[0, 0].bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

	axs[1,1].text(x[0], -0.15*5*7, 'Dual Optimizer', fontsize = 7, horizontalalignment='center')
	axs[1,1].text(x[1], -0.15*5*7, 'Score', fontsize = 7, horizontalalignment='center')
	axs[1,1].text(x[2], -0.15*5*7, 'Hybrid', fontsize = 7, horizontalalignment='center')
	axs[1,1].text(x[3], -0.15*5*7, 'RL', fontsize = 7, horizontalalignment='center')

	# axs[1,0].text((x[1]+x[2])/2, -0.15*10*2, 'Framework', fontsize = 10, horizontalalignment='center')

	plt.sca(axs[1,1])
	plt.xticks([])
	plt.yticks(fontsize=14)
	plt.ylabel("System Load (%)", fontsize=10)
	# axs[1, 1].set_yticks(np.arange(0, 100, 20))
	axs[1, 1].set_yticks(np.arange(0, 100, 15))
	plt.ylim(0,80)
	

	axs[1,1].grid(axis='y', linestyle='--', color='lightgrey')

	figure.savefig('plot-bar-3.jpeg', dpi=300, bbox_inches="tight")
	
	
if __name__ == '__main__':
	main()
