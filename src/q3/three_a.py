import numpy as np
import argparse

class DataLoader:
    def __init__(this, X, Y):
        this.xs = X
        this.ys = Y
        this.current_ind = 0
        this.size = len(X)
    def get_iterator(this):
        this.current_ind = 0

        def _wrapper():
            while this.current_ind < this.size:
                x_i = this.xs[this.current_ind]
                y_i = this.ys[this.current_ind]
                yield (x_i, y_i)
                this.current_ind += 1

        return _wrapper()

class Scaler:
    def __init__(this, mu=0, sig_sq=1):
        this.mu = mu
        this.sig_sq = sig_sq

    def transform(this, data):
        return (data - this.mu) / np.sqrt(this.sig_sq)

    def inverse_transform(this, data):
        return (data * np.sqrt(this.sig_sq)) + this.mu

    def fit(this, data):
        mean = np.zeros(data.shape[1])
        var = np.zeros(data.shape[1])
        n = data.shape[0]
        for x in data:
            mean += x/n
            var += np.multiply(x,x)/(n-1)
        var = var - mean * mean * n/(n-1)
        this.mu = mean
        this.sig_sq = var

def getdata(dx,dy):
    with open(dx,'r') as f:
        _x = f.readlines()
    with open(dy,'r') as f:
        _y = f.readlines()

    data = np.array([])
    label = np.array([])

    for it,(x,y) in enumerate(zip(_x,_y)):
        features = np.array([float(k.strip()) for k in x.strip().split(',')])
        if it == 0:
            data = np.append(data, features)
            data = data.reshape((1,data.shape[0]))
        else:
            data = np.append(data, features.reshape((1,features.shape[0])), axis = 0)
        label = np.append(label,float(y.strip()))

    return (data, label)

def get_dataloader(X, Y):
    loader = dict()
    loader['scaler'] = [Scaler(),Scaler()]
    loader['scaler'][0].fit(X[...,0].reshape((X.shape[0],1)))
    loader['scaler'][1].fit(X[...,1].reshape((X.shape[0],1)))
    X[...,0] = loader['scaler'][0].transform(X[...,0])
    X[...,1] = loader['scaler'][1].transform(X[...,1])
    loader['data'] = DataLoader(X,Y)
    return loader

def dist(a,b):
    return np.sqrt(np.sum(np.square(a-b)))

# IMPLEMENT THE SIGMOID, AND SIGMOID_PRIME FUNCTIONS
def sigmoid(x):
    ##############  FOR STABILIZING THE SIGMOID  ################
    if x >= 0:
        return 1/(1+np.exp(-x))
    else:
        return np.exp(x) / (1+np.exp(x))

def sigmoid_prime(x):
    eps = 1e-40
    #############  FOR STABILIZING THE SIGMOID' #################
    ##########################################^ MEANS PRIME######
    ################################PRIME MEANS DERIVATIVE#######
    if x >= 0:
        ex = np.exp(-x)
    else:
        ex = np.exp(x)
    #############################################################
    Q = ex + 1/(ex+eps) + 2
    return 1/Q

def J(th,X,Y):
    ones = np.ones((X.shape[0],1))
    x_aug = np.concatenate([ones,X],axis=1)
    cost = 0.0
    for x,y in zip(x_aug,Y):
        y_hat = sigmoid(th@x)
        ################  FOR STABILIZING THE LOG  ##################
        if y_hat == 0:
            y_hat += 1e-10
        if y_hat == 1:
            y_hat -= 1e-10
        #############################################################
        cost += y*np.log(y_hat)+(1-y)*np.log(1-y_hat)
    return cost

def three_a(dataloader):
    # STEP 1 INIT
    ths = np.array([0 for _ in range(3)])
    nb_epochs = 200
    eps = 1e-3
    
    Js = []
    for e in range(1,nb_epochs+1):
        grad = np.zeros(ths.shape)
        hess = np.zeros((ths.shape[0],ths.shape[0]))
    # STEP 2 calc gradient:
    # STEP 2.1 start the for loop
        for i, (x, y) in enumerate(dataloader['data'].get_iterator()):
            print("Epoch: [",e,"/",nb_epochs,"]","\tSample number: {:0>3d}".format(i+1),end="\r")
            ones = np.ones((1,))
            x_aug = np.concatenate([ones,x])
    # STEP 2.2 calculate gradient component
            th_T_x = ths@x_aug
            print("Epoch: [",e,"/",nb_epochs,"]","\tSample number: {:0>3d}\t".format(i+1,th_T_x),end="\r")
            y_hat = sigmoid(th_T_x)

            error = (y - y_hat)
            error = np.repeat(error,3)
            grad += np.multiply(error,x_aug)
    # STEP 3 calculate the hessian component
            x_mat = x_aug.reshape((1,x_aug.shape[0]))
            hess -= x_mat.T@x_mat * sigmoid_prime(th_T_x) # x_mat.T@x_mat BECAUSE x_mat is a row vector
    # STEP 4 update parameters
        pr_th = np.array([t for t in ths])
        ths = ths + np.linalg.inv(hess + 1e-1*np.eye(3))@grad
    # STEP 5 check for convergence and repeat until converged
        Js.append(J(ths,dataloader['data'].xs,dataloader['data'].ys))
        if len(Js) > 1 and dist(Js[-2],Js[-1]) < eps:
            print("\n","*"*10," CONVERGED ","*"*10)
            return ths
    return ths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-x",required=True)
    parser.add_argument("--data-y",required=True)
    args = parser.parse_args()
    #############################################################
    x, y = getdata(args.data_x,args.data_y)
    dataloader = get_dataloader(x,y)
    ths = three_a(dataloader)
    print("\n",ths)
    np.save('ths.npy',ths)

if __name__ == "__main__":
    main()
