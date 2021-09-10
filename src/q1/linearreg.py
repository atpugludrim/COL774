import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
def getdata():
    with open('/home/mridul/scai/ml/hw1/data/q1/linearX.csv','r') as f:
        _x = f.readlines()
    with open('/home/mridul/scai/ml/hw1/data/q1/linearY.csv','r') as f:
        _y = f.readlines()

    x = []
    y = []
    mean = 0
    var = 0
    n = 0
    for i,j in zip(_x,_y):
        x.append([float(i.strip()),1.0])
        y.append(float(j.strip()))
        mean += x[-1][0]
        var += x[-1][0]*x[-1][0]
        n += 1
    var = (var - mean)/(n-1)
    mean = mean/n

    for i in range(len(x)):
        x[i][0] = (x[i][0]-mean)/np.sqrt(var)
    return x,y

def J(th,x,y):
    cost = 0.0
    m = len(x)
    for i in range(len(x)):
        h_th_x = 0.0
        for j in range(2):
            h_th_x += th[j] * x[i][j]
        cost += np.square(y[i]-h_th_x)
    cost = cost/(2*m)
    return cost

def norm(a):
    return np.sqrt(np.sum(np.square(np.array(a))))

def linreg(x,y,d=2):
    init = [20,0.5]
    init = [0,0]
    thetas = [k+np.random.rand()*1e-4 for _,k in zip(range(d),init)]
    theta_history = [[t for t in thetas]]
    lr = 1e-1
    nb_epochs = 70000
    nb_epochs = 100
    m = len(x)
    eps = 1e-9

    costs = []
    for e in range(1,nb_epochs+1):
        print("At epoch:",e,','.join(['{:.2f}'.format(t) for t in thetas]))
        costs.append(J(thetas,x,y))
        if len(costs)>1 and (norm(costs[-1]-costs[-2]) < eps):
            print("Converged at epoch ",e)
            break
        grad = [0.0 for _ in range(d)]
        for i in range(m):
            h_th_x = 0.0
            for j in range(d):
                h_th_x += thetas[j] * x[i][j]
            for j in range(d):
                grad[j] += x[i][j] * (y[i] - h_th_x)
        for j in range(d):
            grad[j] = grad[j] / m
            thetas[j] += lr * grad[j]
        theta_history.append([t for t in thetas])
    return thetas, costs, theta_history

def main():
    x,y = getdata()
    ths, Js, thh = linreg(x,y,d=2)
    plt.plot([_ for _ in range(len(Js))],np.array(Js),'C1--')
    plt.show()

    plt.figure()
    x_s = np.array(x)[:,0]
    idx = np.argsort(x_s)
    plt.plot(x_s,y,'C1o',markersize=2)
    th_hat = ths
    y_hat = []
    for k in x:
        y_hat.append(k[0]*th_hat[0]+k[1]*th_hat[1])
    y_hat = np.array(y_hat)
    plt.plot(x_s[idx],y_hat[idx],'C0')
    ax = plt.gca()
    ax.set_xlim(-.05,.15)
    ax.set_ylim(0.85,1.1)
    plt.show()

    #########################################################
    # DRAWING PATH ON CONTOUR PLOT                          #
    #########################################################
    th0 = np.linspace(-30,30,50)
    th1 = np.linspace(0,2,50)
    th0, th1 = np.meshgrid(th0, th1)
    Z = np.empty(th0.shape)
    for i in range(th0.shape[0]):
        for j in range(th1.shape[1]):
            Z[i,j] = J([th0[i,j],th1[i,j]],x,y)
    fig = plt.figure()
    ax = fig.gca()
    cp = ax.contour(th1,th0,Z,30,cmap='RdGy')
    # np.save('thetas.npy',np.array(thh))
    # J_ = []
    for j,t in enumerate(thh):
        if j < 100 or not j % 200:
            ax.plot(t[1],t[0],'gx',markersize=2)
            # J_.append(J(t,x,y))
    # np.save('Js.npy',np.array(J_))
    ax.set_xlabel('th0')
    ax.set_ylabel('th1')
    fig.colorbar(cp)
    plt.show()

main()

#############################################################
# EXTRAS                                                    #
#############################################################

#############################################################
# DRAW THE LOSS SURFACE                                     #
#############################################################

# x, y = getdata()
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# 
# th0 = np.linspace(-30,30,50)
# th1 = np.linspace(0,2,50)
# th0, th1 = np.meshgrid(th0, th1)
# Z = np.empty(th0.shape)
# for i in range(th0.shape[0]):
#     for j in range(th1.shape[1]):
#         Z[i,j] = J([th0[i,j],th1[i,j]],x,y)
# # np.save('Costs.npy',Z)
# surface = ax.plot_surface(th0,th1,Z,cmap=cm.RdGy,rstride=2,cstride=2)
# ax.set_xlabel('th1')
# ax.set_ylabel('th0')
# ax.set_zlabel('J')
# plt.show()

#############################################################
# DRAW THE CONTOUR PLOT                                     #
#############################################################

# fig = plt.figure()
# ax = fig.gca()
# cp = ax.contour(th1,th0,Z,20,cmap='RdGy')
# #cp = ax.contourf(th1,th0,Z,20,cmap='RdGy')
# ax.set_xlabel('th0')
# ax.set_ylabel('th1')
# fig.colorbar(cp)
# plt.show()

#############################################################
# DRAW GRAPHS KEEPING ONE OF THE THETAS CONSTANT            #
#############################################################
# x,y = getdata()
# Js = []
# ths = []
# fig,ax = plt.subplots(2,2)
# for i in range(2):
#     for j in range(2):
#         th0 = 1+np.random.random()*1e-5
#         for th in [k/10 for k in range(-10,15)]:
#             #Js.append(J([0,1],x,y))
#             Js.append(J([th,th0],x,y))
#             ths.append(th)
# 
#         idx = np.argsort(ths)
#         ax[i][j].plot(np.array(ths)[idx],np.array(Js)[idx])
# plt.show()

#############################################################
# HAND PICKED THETA BY LOOKING AT THE GRAPHS                #
#############################################################
# 
# plt.figure()
# x_s = np.array(x)[:,0]
# idx = np.argsort(x_s)
# plt.plot(x_s,y,'C1o',markersize=2)
# th_opt = [0,1]
# y_opt = []
# for k in x:
#     y_opt.append(k[0]*th_opt[0]+k[1]*th_opt[1])
# y_opt = np.array(y_opt)
# plt.plot(x_s[idx],y_opt[idx],'C0')
# ax = plt.gca()
# ax.set_xlim(-.05,.15)
# ax.set_ylim(0.85,1.1)
# plt.show()
