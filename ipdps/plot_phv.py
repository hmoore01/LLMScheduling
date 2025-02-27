import numpy as np
import matplotlib.pyplot as plt

def main():
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)
    x = np.arange(25)
    y = np.array([1.88, 2.1475, 1.88, 2.23899, 2.24101, 1.88, 2.23827, 2.24935, 2.26535, 2.25223, 2.26683, 2.23896, 2.23931, 2.23931, 2.23955, 2.24111, 2.23971, 2.24121, 2.24121, 2.24336, 2.24371, 2.24162, 2.2515, 2.24489, 2.24718])
    ax.plot(x, y, c ="green", label='Azure', marker='o')
    
    the_list = []
    for i in range(10):
        the_list.append(2.7*i)
    x = np.array(the_list)
    y = np.array([4.53814, 4.90334, 1.89413, 4.95816, 1.89406, 5.06695, 5.05895, 5.05895, 5.06819, 5.07019])
    ax.plot(x, y, c ="green", label='Huawei', marker='*')
    
    ax.legend()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = ['2', '4', '6', '8', '10', '12', '14']
    ax.set_xticklabels(labels)
    plt.ylabel("PHV Value of Pareto Front ($10^{14}$)", fontsize=14)
    plt.xlabel("Optimization Time (minute)", fontsize=14)
    fig.savefig('phv-1.jpeg', dpi=1200, bbox_inches="tight")
    
    
if __name__ == '__main__':
    main()