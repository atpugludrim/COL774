import matplotlib.pyplot as plt
import numpy as np
import pickle
from four_a import getdata, Scaler
from scipy import stats
import sys

def main():
    with open('th.pkl','rb') as f:
        th = pickle.load(f)
    with open('th_2.pkl','rb') as f:
        th_2 = pickle.load(f)
    x,y = getdata()
    #############################################################
    if len(sys.argv) > 1 and sys.argv[1] == "-scale":
        s0 = Scaler()
        s1 = Scaler()

        s0.fit(x[...,0].reshape((100,1)))
        s1.fit(x[...,1].reshape((100,1)))

        x[...,0] = s0.transform(x[...,0])
        x[...,1] = s1.transform(x[...,1])
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
    #############################################################
    ind_0 = np.where(y==0)
    x_0 = x[ind_0]

    f1_0_min = min(x_0[...,0])
    f1_0_max = max(x_0[...,0])

    f2_0_max = max(x_0[...,1])
    f2_0_min = min(x_0[...,1])

    f1 = np.linspace(f1_0_min,f1_0_max,20)
    f2 = np.linspace(f2_0_min,f2_0_max,20)

    f1, f2 = np.meshgrid(f1,f2)
    z = np.dstack((f1,f2))
    rv = stats.multivariate_normal(th_2['mu0'],th_2['sig0'])

    plt.contour(f1,f2,rv.pdf(z),20,cmap='Pastel1')

    ind_1 = np.where(y==0)
    x_1 = x[ind_1]

    f1_1_min = min(x_1[...,0])
    f1_1_max = max(x_1[...,0])

    f2_1_max = max(x_1[...,1])
    f2_1_min = min(x_1[...,1])

    f1 = np.linspace(f1_1_min,f1_1_max,20)
    f2 = np.linspace(f2_1_min,f2_1_max,20)

    f1, f2 = np.meshgrid(f1,f2)
    z = np.dstack((f1,f2))
    rv = stats.multivariate_normal(th_2['mu1'],th_2['sig1'])

    plt.contour(f1,f2,rv.pdf(z),20,cmap='plasma')
    plt.legend()
    plt.show()
main()
