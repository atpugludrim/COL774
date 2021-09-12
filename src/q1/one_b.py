import numpy as np
import matplotlib.pyplot as plt
import argparse
from one_a import getdata, Scaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-x',required=True)
    parser.add_argument('--data-y',required=True)
    args = parser.parse_args()
    #############################################################
    X, Y = getdata(args.data_x,args.data_y)
    scaler = Scaler()
    scaler.fit(X)
    with open('ths_ref.npy','rb') as f:
        ths = np.load(f)

    X = scaler.transform(X)

    x_min = np.min(X)
    x_max = np.max(X)
    x_interval = [x_min, x_max]
    #x_interval = scaler.transform([x_min, x_max])
    #print(x_interval)
    y_min = ths[0] + x_interval[0]*ths[1]
    y_max = ths[0] + x_interval[1]*ths[1]

    for x, y in zip(X,Y):
        plt.plot(x, y, 'C2o', markersize=2)
    plt.plot([x_min,x_max],[y_min,y_max],'C0--',label='Hypothesis')
    plt.legend()
    ax = plt.gca()
    ax.set_ylim(0,1.5)
    #plt.savefig('one_b.png')
    plt.show()

if __name__=="__main__":
    main()
