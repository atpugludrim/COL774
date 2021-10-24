import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import *

def timeit(func):
    def inner(*args):
        s = time.time()
        r = func(*args)
        e = time.time()
        print("Runtime \"{}\": {:.2f}s or {:.2f}min".format(func.__name__,e-s,(e-s)/60))
        return r
    return inner

class DataLoader:
    def __init__(this, xs, ys, bs):
        if type(xs) == np.ndarray:
            this.xs = xs
        else:
            this.xs = np.array(xs)
        if type(ys) == np.ndarray:
            this.ys = ys
        else:
            this.ys = np.array(ys)
        this.bs = bs
    def shuffle(this):
        ind = np.random.permutation(this.xs.shape[0])
        this.xs = this.xs[ind]
        this.ys = this.ys[ind]
    def get_iterator(this):
        end_ind = 0
        def _iter():
            nonlocal end_ind
            while end_ind < this.xs.shape[0]:
                start_ind = end_ind
                end_ind = min(end_ind+this.bs,this.xs.shape[0])
                yield this.xs[start_ind:end_ind,...], this.ys[start_ind:end_ind,...]
        return _iter()

def make_mlp_model(n,n_neurons,r,X,y):
    W = [variable(np.random.randn(n,n_neurons[0]))]
    b = [variable(np.random.randn(n_neurons[0]))]
    h = [sigmoid(add(matmul(X,W[-1]),b[-1]))]
    for idx in range(1,len(n_neurons)):
        W.append(variable(np.random.randn(n_neurons[idx-1],n_neurons[idx])))
        b.append(variable(np.random.randn(n_neurons[idx])))
        h.append(sigmoid(add(matmul(h[-1],W[-1]),b[-1])))
    W.append(variable(np.random.randn(n_neurons[-1],r)))
    b.append(variable(np.random.randn(r)))
    # h.append(softmax(add(matmul(h[-1],W[-1]),b[-1])))
    h.append(sigmoid(add(matmul(h[-1],W[-1]),b[-1])))
    return W,b,h

def test(x_test, y_test, den):#, W, b):
    global W_vals, b_vals
    def sigmoid_(z):
        return 1/(1+np.exp(-z))
    def softmax_(z):
        return np.exp(z)/np.sum(np.exp(z),axis=1)[:,None]
    h = x_test
    for w,b in zip(W_vals[:-1],b_vals[:-1]):
        h = sigmoid_(h@w+b)
    o = sigmoid_(h@W_vals[-1]+b_vals[-1])
    # o = softmax_(h@W_vals[-1]+b_vals[-1])
    y_hat = np.argmax(o,axis=1)
    y_test_ = np.argmax(y_test,axis=1)
    return np.sum(y_hat==y_test_)/den

@timeit
def main():
    global W_vals, b_vals
    df = pd.read_csv('train.csv',header=None)
    x_train = df.iloc[:,:-10].values
    y_train = df.iloc[:,-10:].values

    df_val = pd.read_csv('val.csv',header=None)
    x_val = df.iloc[:,:-10].values
    y_val = df.iloc[:,-10:].values
    den = len(x_val)

    bs = 100
    dl = DataLoader(x_train, y_train, bs)
    dl.shuffle()

    X = placeholder()
    Y = placeholder()
    W, b, h = make_mlp_model(85,[100],10,X,Y)
    y_hat = h[-1]
    # J = negative(reduce_sum(reduce_sum(multiply(Y,log(y_hat)),axis=1)))
    m2 = constant(1/(2*bs))
    J = multiply(m2,reduce_sum(reduce_sum(square(add(Y,negative(y_hat))),axis=1)))
    # minimization_op = SGDOptimizer(lr=0.003).minimize(J)
    minimization_op = SGDOptimizer(lr=0.1).minimize(J)

    session = Session()

    js = []
    ns = 33
    converged = False
    epoch = 1000
    eps = 1e-30
    # val_acc = 0
    # patience = 0
    # max_patience = bs
    # W_best, b_best = None, None
    for step in range(1,1+epoch):
        for i, (x, y) in enumerate(dl.get_iterator()):
            feed_dict = {X:x,Y:y}
            J_val = session.run(J, feed_dict) # forward propagation
            # js.append(J_val/bs)
            js.append(J_val)
            if (step == 1 or not step % 100) and i == 1:
                print("Step:",step,"loss:",J_val)# normalize by batch size
            #############################################################
            # W_vals = [w.val for w in W]
            # b_vals = [b_.val for b_ in b]
            # val_accuracy = test(x_val, y_val, den)
            # if val_accuracy > val_acc:
            #     val_acc = val_accuracy
            #     patience = 0
            #     W_best = [w for w in W_vals]
            #     b_best = [b_ for b_ in b_vals]
            # else:
            #     patience += 1
            # if patience == max_patience:
            #     converged = True
            #     break
            #############################################################
            if len(js) >= 2*ns:
                j1 = np.mean(js[-2*ns:-ns])
                j2 = np.mean(js[-ns:])
                if np.sqrt(np.sum(np.square(j1-j2))) < eps:
                    converged = True
                    break
            session.run(minimization_op,feed_dict) # backward propagation and grad subtraction
        if converged == True:
            print("","*"*10,"converged","*"*10)
            break

    df = pd.read_csv('test.csv',header=None)
    x_test = df.iloc[:,:-10].values
    y_test = df.iloc[:,-10:].values
    den = len(x_test)

    W_vals = [w.val for w in W]
    b_vals = [b_.val for b_ in b]
    end_ind = 0
    acc = 0
    while end_ind < den:
        s_ind = end_ind
        end_ind = min(s_ind+4000,den)
        x_test_slice = x_test[s_ind:end_ind]
        y_test_slice = y_test[s_ind:end_ind]
        acc += test(x_test_slice, y_test_slice,den)
    print(acc*100)

    # W_vals = [w for w in W_best]
    # b_vals = [b_ for b_ in b_best]
    # end_ind = 0
    # acc = 0
    # while end_ind < den:
    #     s_ind = end_ind
    #     end_ind = min(s_ind+4000,den)
    #     x_test_slice = x_test[s_ind:end_ind]
    #     y_test_slice = y_test[s_ind:end_ind]
    #     acc += test(x_test_slice, y_test_slice,den)
    # print(acc*100)

if __name__=="__main__":
    main()
