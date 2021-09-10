#############################################################
# DIDN'T REALLY WORK, THE OUTPUT DIDN'T COME OUT AS I EXPE- #
# CTED IT TO BE. WANTED TO TRY WITH NONLINEAR DATA NEXT.    #
#############################################################
import numpy as np
import matplotlib.pyplot as plt

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
        x.append([1.0,float(i.strip())])
        y.append(float(j.strip()))
        mean += x[-1][1]
        var += x[-1][1]*x[-1][1]
        n += 1
    var = (var - mean)/(n-1)
    mean = mean/n

    for i in range(len(x)):
        x[i][1] = (x[i][1]-mean)/var
    return np.array(x),np.array(y),mean,var

def dist(a,b):
    return np.sqrt(np.sum(np.square(a-b)))

def wi(x, x_i, tau_sq):
    return np.exp(-np.sum(np.square(x-x_i))/(2*tau_sq))

def delJ(th, x, y, X, tau_sq = 100):
    h_th_x = lambda a, b: a[0] * b[0] + a[1] * b[1]
    delJ = np.empty(x.shape[1])
    m = len(y)

    for x_i, y_i in zip(x,y):
        residual = 1/m*wi(X, x_i, tau_sq)*(y_i - h_th_x(x_i, th))
        for j in range(len(x_i)):
            delJ[j] += x_i[j]*residual
    return delJ

def regress(x, y, X):
    th = np.random.rand(2,)*0.1
    lr = 1e-2
    nb_epochs = 300
    eps = 1e-9

    for e in range(1,nb_epochs+1):
        print("Epoch: [{}/{}] around x = {:.3f}".format(e,nb_epochs,X))
        pr_th = np.array([i for i in th])
        dels = delJ(th,x,y,X,tau_sq=100000)
        for j, del_thj in enumerate(dels):
            th[j] += lr * del_thj
        if dist(pr_th, th) < eps:
            print("Converged ...")
            break

    return th

def make_curve(x_min, x_max, nbpoints):
    x_train, y_train, mean, var = getdata()
    x_values = np.linspace(x_min, x_max, nbpoints)
    y_values = np.empty(x_values.shape)

    h = lambda x, t: x[0]*t[0]+x[1]*t[1] 
    for idx, x in enumerate(x_values):
        th = regress(x_train, y_train, (x-mean)/var)
        y_values[idx] = h([1, (x-mean)/var], th)
    
    for x_i, y_i in zip(x_train, y_train):
        plt.plot(x_i[1],y_i,'C1o',markersize=2)
    plt.plot(x_values, y_values)
    plt.show()

make_curve(5,11,50)
