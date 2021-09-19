import numpy as np
import matplotlib.pyplot as plt

def getdata(s=100,mu1=25,sig1=25,noise_sig=25):
    ths = [5,0.1,0.1,0.0001,1]
    x0 = np.ones(s)
    x1 = np.random.randn(s)*sig1+mu1
    x2 = np.square(x1)
    x3 = np.power(x1,3)
    eps = np.random.randn(s)*noise_sig
    xs = np.stack([x0,x1,x2,x3,eps])
    y = ths@xs
    return np.stack([x0,(x1-mu1)/sig1]).T,y,mu1,sig1

def dist(a,b):
    return np.sqrt(np.sum(np.square(a-b)))

def wi(x, x_i, tau_sq):
    return np.exp(-np.sum(np.square(x-x_i))/(2*tau_sq))

def delJ(th, x, y, X, tau_sq = 100):
    h_th_x = lambda a, b: a[0] * b[0] + a[1] * b[1]
    delJ = np.zeros(x.shape[1])
    m = len(y)

    for x_i, y_i in zip(x,y):
        residual = 1/m*wi(X, x_i[1], tau_sq)*(y_i - h_th_x(x_i, th))
        for j in range(len(x_i)):
            delJ[j] += x_i[j]*residual
    return delJ

def regress(x, y, X, idx):
    th = np.random.rand(2,)*0.1
    lr = 0.5e-1
    nb_epochs = 200
    eps = 1e-7

    for e in range(1,nb_epochs+1):
        print("Pt. {}\tEpoch: [{}/{}] around x = {:.3f}, th = {:.4f}, {:.4f}".format(idx,e,nb_epochs,X,th[0],th[1]),end="\r")
        if np.isnan(X):
            raise Exception("Idk what")
        pr_th = np.array([i for i in th])
        dels = delJ(th,x,y,X,tau_sq=2)
        for j, del_thj in enumerate(dels):
            th[j] += lr * del_thj
        if dist(pr_th, th) < eps:
            print("\nConverged ...",end="\r")
            break
    print("")

    return th

def make_curve(x_min, x_max, nbpoints):
    x_train, y_train, mean, sig = getdata()
    x_values = np.linspace(x_min, x_max, nbpoints)
    y_values = np.zeros(x_values.shape)

    h = lambda x, t: x[0]*t[0]+x[1]*t[1] 
    for idx, x in enumerate(x_values):
        th = regress(x_train, y_train, (x-mean)/sig, idx+1)
        y_values[idx] = h([1, (x-mean)/sig], th)
     
    plt.plot(x_train[:,1]*sig+mean,y_train,color='C3',alpha=0.3,markersize=5,marker='H',linestyle="",label="data")
    plt.plot(x_values, y_values,marker='p',markersize=5,alpha=0.5,color='C2',linestyle="--",label='hypothesis')
    plt.legend()
    plt.show()

total_points=20
print("total points:",total_points)
make_curve(-50,100,total_points)
