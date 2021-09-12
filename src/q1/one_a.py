import numpy as np
import argparse

class DataLoader:
    def __init__(this, X, Y):
        this.size = len(X)
        this.xs = X
        this.ys = Y
        this.current_ind = 0
    def get_iterator(this):
        this.current_ind = 0

        def _wrapper():
            while this.current_ind < this.size:
                x_i = this.xs[this.current_ind, ...]
                y_i = this.ys[this.current_ind, ...]
                yield (x_i, y_i)
                this.current_ind += 1
        return _wrapper()

class Scaler:
    def __init__(this, mu=0, sig_sq=1):
        this.mu = mu
        this.sig_sq = sig_sq
    def fit(this, data):
        mean = np.zeros(data.shape[1])
        var = np.zeros(data.shape[1])
        n = data.shape[0]
        for x in data:
            mean += x/n
            var += np.multiply(x,x)/(n-1)
        var = var - mean*mean*n/(n-1)
        this.mu = mean
        this.sig_sq = var
    def transform(this, data):
        return (data - this.mu) / np.sqrt(this.sig_sq)
    def inverse_transform(this, data):
        return (data * np.sqrt(this.sig_sq)) + this.mu

def getdata(dx,dy):
    with open(dx,'r') as f:
        _x = f.readlines()
    with open(dy,'r') as f:
        _y = f.readlines()
    data = np.array([])
    labels = np.array([])
    for x, y in zip(_x, _y):
        data = np.append(data,float(x.strip()))
        labels = np.append(labels,float(y.strip()))
    return data.reshape((data.shape[0],1)), labels

def get_dataloader(X, Y):
    loader = dict()
    loader['scaler'] = [Scaler()]
    loader['scaler'][0].fit(X)
    X[..., 0] = loader['scaler'][0].transform(X[...,0])
    loader['data'] = DataLoader(X,Y)
    return loader

def dist(a, b):
    # DISTANCE METRIC INCURRED BY EUCLIDEAN NORM #
    return np.sqrt(np.sum(np.square(a-b)))

def J(th, X, Y):
    X = X.reshape((X.shape[0],1))
    ones = np.ones((X.shape[0],1))
    x_aug = np.concatenate([ones,X],axis=1)
    return np.add.reduce(1/(2*Y.shape[0])*np.square(Y-th@x_aug.T)) # X@Y means matrix multiplication of X and Y in NP lingo

def one_a(dataloader):
    lr = 1*1e-3
    ths = np.array([0.0 for _ in range(2)])
    #ths = np.array([np.random.rand()*1e-1 for _ in range(2)])
    history = [[t for t in ths]]
    nb_epochs = 4000
    m = dataloader['data'].size
    eps = 1e-6

    Js = np.array([])
    for e in range(1,nb_epochs+1):
        grad = np.zeros(ths.shape)
        for i, (x, y) in enumerate(dataloader['data'].get_iterator()):
            print("Epoch: [",e,"/",nb_epochs,"]","\tSample number: {:0>3d}".format(i),end="\r")
            ones = np.ones((x.shape[0],1))
            x_aug = np.concatenate([ones,x.reshape((x.shape[0],1))],axis=1)
            y_hat = ths@x_aug.T

            error = (y - y_hat) / m
            error = np.repeat(error.reshape((error.shape[0],1)), 2, axis = 1) # FOR THE ELEMENTWISE MULTIPLICATION
            grad += np.add.reduce(np.multiply(error,x_aug),axis = 0) # NP.ADD.REDUCE IS THE SAME AS NP.SUM BUT FASTER

        ths += lr * grad
        history.append([t for t in ths])

        Js = np.append(Js, J(ths, dataloader['data'].xs, dataloader['data'].ys))
        if Js.shape[0] > 1 and dist(Js[-2],Js[-1]) < eps:
            print("\n","*"*10," CONVERGED ","*"*10)
            return ths, Js, np.array(history)
    return ths, Js, np.array(history)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-x',required=True)
    parser.add_argument('--data-y',required=True)
    args = parser.parse_args()
    #############################################################
    X, Y = getdata(args.data_x,args.data_y)
    dataloader = get_dataloader(X, Y)
    ths, Js, thh = one_a(dataloader)
    print("\nThetas are: ",ths)
    np.save("ths_ref.npy",ths)
    np.save("Js_ref.npy",Js)
    np.save("th_history_ref.npy",thh)

if __name__=="__main__":
    main()
