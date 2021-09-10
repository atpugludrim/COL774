import numpy as np
import copy
import matplotlib.pyplot as plt
import argparse
from three_a import getdata, get_dataloader

def main():
    ths = np.load('ths.npy')
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-x",required=True)
    parser.add_argument("--data-y",required=True)
    args = parser.parse_args()
    #############################################################
    x, y = getdata(args.data_x,args.data_y)
    x2 = copy.deepcopy(x)
    y2 = copy.deepcopy(y)
    dl = get_dataloader(x2,y2)
    x_min = min(x[...,0])
    x_max = max(x[...,0])
    x_transformed = dl['scaler'][0].transform(np.array([x_min,x_max]))
    y_min = -(ths[0]+ths[1]*x_min)/ths[2]
    y_max = -(ths[0]+ths[1]*x_max)/ths[2]
    for p_x, p_y in zip(x,y):
        if p_y == 0:
            plt.plot(p_x[0],p_x[1],color='C2',marker=10)
        else:
            plt.plot(p_x[0],p_x[1],color='C5',marker=11)
    plt.plot([x_min,x_max],[y_min,y_max],'C0--',label='Separator')
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.show()

if __name__ == "__main__":
    main()
