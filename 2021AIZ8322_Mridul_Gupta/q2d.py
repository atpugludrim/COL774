import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import *

class DummyFile:
    def write(this,s):pass
    def flush(this,*args):pass

sys.stdout = DummyFile() # silencing all outputs
sys.stderr = DummyFile() # silencing all error outputs
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

def test(x_test, y_test, W_vals, b_vals):
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
    return y_hat, y_test_

def train(hla,bs,lr,epoch,eps):
    global x_test, y_test
    X = placeholder()
    Y = placeholder()
    W, b, h = make_mlp_model(85,hla,10,X,Y)
    y_hat = h[-1]
    # J = negative(reduce_sum(reduce_sum(multiply(Y,log(y_hat)),axis=1)))
    m2 = constant(1/(2*bs))
    J = multiply(m2,reduce_sum(reduce_sum(square(add(Y,negative(y_hat))),axis=1)))
    opt = SGDOptimizer(lr=lr)
    minimization_op = opt.minimize(J)

    session = Session()
    js = []
    ns = bs
    converged = False
    s = time.time()
    for step in range(1,1+epoch):
        opt.lr = lr/np.sqrt(step)
        for i, (x, y) in enumerate(dl.get_iterator()):
            feed_dict = {X:x,Y:y}
            J_val = session.run(J, feed_dict) # forward propagation
            js.append(J_val)
            if (step == 1 or not step % 100) and i == 1:
                print("Step:",step,"loss:",J_val)# normalize by batch size
            if len(js) >= 2*ns:
                j1 = np.mean(js[-2*ns:-ns])
                j2 = np.mean(js[-ns:])
                if max(j1-j2,j2-j1) < eps:
                    converged = True
                    break
            session.run(minimization_op,feed_dict) # backward propagation and grad subtraction
        if converged == True:
            print("","*"*10,"converged","*"*10)
            break
    print(step,"steps performed")
    e = time.time()

    den = len(x_test)

    W_vals = [w.val for w in W]
    b_vals = [b_.val for b_ in b]
    end_ind = 0
    acc = 0
    y_hats, y_trus = np.array([]), np.array([])
    while end_ind < den:
        s_ind = end_ind
        end_ind = min(s_ind+4000,den)
        x_test_slice = x_test[s_ind:end_ind]
        y_test_slice = y_test[s_ind:end_ind]
        y_hat, y_tru = test(x_test_slice, y_test_slice, W_vals, b_vals)
        acc += np.sum(y_hat==y_tru)/den
        y_hats = np.append(y_hats, y_hat)
        y_trus = np.append(y_trus, y_tru)
    print(acc*100)
    return acc*100, e-s, y_hats, y_trus

def make_conf_mat(y_hat, y, filename):
    fig = plt.figure()
    cm = [[0 for _ in range(10)] for _ in range(10)]
    for t,p in zip(y,y_hat):
        cm[int(t)][int(p)] += 1
    rowLabs = ["Actual {}  ".format(i) for i in range(10)]
    colLabs = ["P {}".format(i) for i in range(10)]
    rcolors = plt.cm.BuPu(np.full(len(rowLabs),0.1))
    ccolors = plt.cm.BuPu(np.full(len(colLabs),0.1))
    celtext = []
    for r in cm:
        celtext.append([str(c) for c in r])
    table = plt.table(cellText=celtext,
            rowLabels=rowLabs,
            colLabels=colLabs,
            cellLoc='center',
            loc='upper left',
            rowColours=rcolors,
            colColours=ccolors,
            )
    table.scale(1,2)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.gca().set_axis_off()
    # plt.show()
    plt.savefig('{}'.format(filename))
    plt.close(fig)
@timeit
def main():
    global dl, x_test, y_test
    df = pd.read_csv('train.csv',header=None)
    x_train = df.iloc[:,:-10].values
    y_train = df.iloc[:,-10:].values

    bs = 100
    dl = DataLoader(x_train, y_train, bs)
    dl.shuffle()

    df = pd.read_csv('test.csv',header=None)
    x_test = df.iloc[:,:-10].values
    y_test = df.iloc[:,-10:].values

    hlas = [_ for _ in range(5,26,5)]
    # hlas.append(100)

    times, acc, = [], []
    for hla in hlas:
        a, t, y_hat, y_tru = train([hla],bs,0.1,1400,-float('inf'))
        times.append(t)
        acc.append(a*100)
        make_conf_mat(y_hat, y_tru, "hidden_{}.png".format(hla))
    fig=plt.figure()
    plt.plot(hlas,times,label='times')
    plt.savefig('times.png')
    plt.close(fig)
    fig=plt.figure()
    plt.plot(hlas,acc,label='acc')
    plt.savefig('acc.png')
    plt.close(fig)

if __name__=="__main__":
    main()
