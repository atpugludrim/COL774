import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
from four_a import getdata, Scaler

def decision_boundary(th,x_min,x_max):
    phi = th['phi']
    mu0 = th['mu0']
    mu1 = th['mu1']
    sig = th['sig']

    sig_inv = np.linalg.inv(sig)

    c = np.log((1-phi)/phi)
    d = (mu0 - mu1).reshape((2,1)) # MAKING A NP VECTOR INTO A MATRIX
    mu0_mat = mu0.reshape((2,1))
    mu1_mat = mu1.reshape((2,1))

    c -= 0.5 * (mu0_mat.T@sig_inv@mu0_mat - mu1_mat.T@sig_inv@mu1_mat)
    ths = d.T@sig_inv

    f = lambda x : -(c+ths[0][0]*x)/ths[0][1]

    y_min = f(x_min)
    y_max = f(x_max)

    return y_min.item(), y_max.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-x',required=True)
    parser.add_argument('--data-y',required=True)
    parser.add_argument('-scale',default=False,action='store_true')
    args = parser.parse_args()
    #############################################################
    with open('th.pkl','rb') as f:
        th = pickle.load(f)
    x,y = getdata(args.data_x,args.data_y)
    if args.scale:
    #############################################################
        s0 = Scaler()
        s1 = Scaler()

        s0.fit(x[...,0].reshape((100,1)))
        s1.fit(x[...,1].reshape((100,1)))

        x[...,0] = s0.transform(x[...,0])
        x[...,1] = s1.transform(x[...,1])
    #############################################################
    
    x1,x2 = min(x[:,0]),max(x[:,0])
    y1,y2 = decision_boundary(th,x1,x2)
    plt.plot([x1,x2],[y1,y2],color='dodgerblue',linestyle='--',lw=0.8,label='Decision bondary')
    flags = [True,True]
    for p,l in zip(x,y):
        if l == 0:
            c = 'C3'
            m=10
            ms = 5
            lab = 'Alaska'
        else:
            c = 'C2'
            m = 11
            ms = 4
            lab = 'Canada'
        if flags[int(l)]:
            plt.plot(p[0],p[1],marker=m, linestyle='',color=c,markersize=ms,label=lab)
            flags[int(l)] = False
        else:
            plt.plot(p[0],p[1],marker=m,color=c,markersize=ms)

    plt.legend()
    ax = plt.gca()
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    #plt.savefig('four_c.png')
    plt.show()

if __name__ == "__main__":
    main()
