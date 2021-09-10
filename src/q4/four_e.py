import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
from four_a import getdata, Scaler
from four_b import decision_boundary

def quadratic_decision_boundary(th,x_coord,lims):
    phi = th['phi']
    mu0 = th['mu0']
    mu1 = th['mu1']
    sig0 = th['sig0']
    sig1 = th['sig1']

    mu0_mat = mu0.reshape((2,1))
    mu1_mat = mu1.reshape((2,1))

    sig0_det = np.linalg.det(sig0)
    sig1_det = np.linalg.det(sig1)

    sig0_inv = np.linalg.inv(sig0)
    sig1_inv = np.linalg.inv(sig1)

    sig_inv_0_minus_1 = sig0_inv - sig1_inv
    mu_0_minus_1 = mu0_mat.T@sig0_inv - mu1_mat.T@sig1_inv

    k = np.log((1-phi)/phi)+0.5*np.log(sig1_det/sig0_det)
    a = k - 0.5 * (mu0_mat.T@sig0_inv@mu0_mat - mu1_mat.T@sig1_inv@mu1_mat)
    b = 2 * a.item()

    alpha = sig_inv_0_minus_1[0,0].item()
    beta = sig_inv_0_minus_1[0,1].item()
    gamma = sig_inv_0_minus_1[1,0].item()
    delta = sig_inv_0_minus_1[1,1].item()

    varphi = mu_0_minus_1[0,0].item()
    pi = mu_0_minus_1[0,1].item()

    A = -delta
    B = 2*pi - (gamma+beta)*x_coord
    C = 2*varphi*x_coord-alpha*x_coord*x_coord+b

    D = np.sqrt(B*B-4*A*C)
    y1 = (-B+D)/(2*A)
    y2 = (-B-D)/(2*A)

    if y1 < lims[0] or y1 > lims[1]:
        y1 = float('Inf')
    if y2 < lims[0] or y2 > lims[1]:
        y2 = float('Inf')

    return [y1,y2]

def main():
    with open('th.pkl','rb') as f:
        th = pickle.load(f)
    with open('th_2.pkl','rb') as f:
        th_2 = pickle.load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-x',required=True)
    parser.add_argument('--data-y',required=True)
    parser.add_argument('-scale',default=False,action='store_true')
    args = parser.parse_args()
    #############################################################
    x,y=getdata(args.data_x,args.data_y)
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
    plt.plot([x1,x2],[y1,y2],color='dodgerblue',linestyle=(0,(7,3,4,3)),lw=1.3,label='Linear decision boundary')
    ############  PLOTTING QUADRATIC BOUNDARY  ##################
    xs = np.linspace(x1,x2,100)
    ys = [quadratic_decision_boundary(th_2,X,[y1,y2]) for X in xs]

    actual_pxs = []
    actual_pys = []
    for p_x, p_y in zip(xs,ys):
        for idx in range(len(p_y)):
            if p_y[idx] != float('Inf'):
                actual_pxs.append(p_x)
                actual_pys.append(p_y[idx])
    plt.plot(actual_pxs,actual_pys,linestyle=(0,(3,1,1,1,1,1)),lw=1.5,color='darkslategrey',label='Quadratic decision boundary')
    #############################################################
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
    #plt.savefig('four_e.png')
    plt.show()

if __name__ == "__main__":
    main()
