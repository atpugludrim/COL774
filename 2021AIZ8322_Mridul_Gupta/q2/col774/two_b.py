import numpy as np
from two_a import two_a
import argparse
import sys

class DataLoader:
    def __init__(this, X, Y, batch_size):
        this.batch_size = batch_size
        this.current_ind = 0
        this.size = len(X)
        this.num_batch = int(this.size // this.batch_size)
        this.xs = X
        this.ys = Y
    def shuffle(this):
        perm = np.random.permutation(this.size)
        xs, ys = this.xs[perm], this.ys[perm]
        this.xs = xs
        this.ys = ys
    def get_iterator(this):
        this.current_ind = 0

        def _wrapper():
            while this.current_ind < this.num_batch:
                start_ind = this.batch_size * this.current_ind
                end_ind = min(this.size, this.batch_size * (this.current_ind + 1))
                x_i = this.xs[start_ind: end_ind, ...]
                y_i = this.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                this.current_ind += 1
        return _wrapper()

class Scaler:
    def __init__(this, mu, sig):
        this.mu = mu
        this.sig = sig
    
    def transform(this, data):
        return (data - this.mu)/this.sig

    def inverse_transform(this, data):
        return (data * this.sig) + this.mu

def getdata(size):
    X, Y = two_a([3.0,1.0,2.0], size, 3.0, 4.0, -1.0, 4.0, 2.0)
    # X, Y = two_a([3.0,1.0,2.0], size, 3.0, 4.0, -1.0, 4.0, 1e-5)
    return X, Y

def get_dataloader(X, Y, bs):
    loader = dict()
    # loader['scaler'] = [Scaler(X[...,0].mean(),X[...,0].std()),Scaler(X[...,1].mean(),X[...,1].std())]
    loader['scaler'] = [Scaler(0,1),Scaler(0,1)]
    X[...,0] = loader['scaler'][0].transform(X[...,0])
    X[...,1] = loader['scaler'][1].transform(X[...,1])
    loader['data'] = DataLoader(X,Y,bs)
    return loader

def dist(a,b):
    return np.sqrt(np.sum(np.square(a-b)))

def J(th, X, Y):
    ones = np.ones((X.shape[0],1))
    x_aug = np.concatenate([ones,X],axis=1)
    return np.add.reduce(1/(2*Y.shape[0])*np.square(Y-th@x_aug.T))

def two_b(dataloader,eps,nb_epochs):
    # STEP 1 init
    lr = 1e-3
    ths = np.array([0.0 for _ in range(3)])
    th_history = [[t for t in ths]]
    m = dataloader['data'].batch_size
    total = dataloader['data'].num_batch

    dataloader['data'].shuffle()
    Js = np.array([])
    for e in range(1,nb_epochs+1):
        for b, (x, y) in enumerate(dataloader['data'].get_iterator()):
            print("Epoch:",e,"\tBatch: ",b+1,end="\r")
            # print("\tData:",x.shape,y.shape)

            ones = np.ones((x.shape[0],1))
            x_aug = np.concatenate([ones,x],axis=1)

            # print("\t\tAugmented:",x_aug.shape)
            # print("\t\tths",ths.shape)

            y_hat = ths@x_aug.T

            # print("\t\ty_hat:",y_hat.shape)
    # STEP 2 calc grad wrt current theta on batch
            # print("\t\t",(y-y_hat).shape)

            error = (y - y_hat) / m
            error = np.repeat(error.reshape((error.shape[0],1)), 3, axis = 1)
            grad = np.add.reduce(np.multiply(error, x_aug),axis=0)

            # print("\t\tgrad:",grad.shape)
    # STEP 3 update theta
            ths += lr * grad
            th_history.append([t for t in ths])
    # STEP 4 check for convergence
            Js = np.append(Js, J(ths, x, y))
            if Js.shape[0] > 999 and not Js.shape[0] % 500:
                if dist(np.mean(Js[-1000:-500]),np.mean(Js[-500:])) < eps:
                    print("\n","*"*10," CONVERGED ","*"*10)
                    return ths, Js, np.array(th_history)
    return ths, Js, np.array(th_history)

def main():
    #############################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs')
    parser.add_argument('--batch-size')
    parser.add_argument('--eps')
    args = parser.parse_args()
    #############################################################
    X, Y = getdata(1e6)

    batch_size = int(2*1e3)
    suffix = ""
    eps = 1e-6
    nb_epochs = 500
    if args.epochs is not None:
        nb_epochs = int(args.epochs)
    if args.batch_size is not None:
        batch_size = int(float(args.batch_size))
        suffix = "_bs{}".format(batch_size)
    if args.eps is not None:
        eps = float(args.eps)
    #############################################################
    dataloader = get_dataloader(X, Y, batch_size)
    ths, Js, thh = two_b(dataloader,eps,nb_epochs)
    print("\nThetas are:",ths)
    np.save("x.npy",X)
    np.save("y.npy",Y)
    np.save('ths{}.npy'.format(suffix),ths)
    np.save('thh{}.npy'.format(suffix),thh)
    np.save('Js{}.npy'.format(suffix),Js)


if __name__ == "__main__":
    main()
