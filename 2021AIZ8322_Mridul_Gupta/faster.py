import numpy as np
import pandas as pd

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

def sigmoid(z):
    return 1/(1+np.exp(z))
def init(shape):
    return np.ones(shape)
    # return np.random.randn(*shape)
def make(hla,n,r):
    if not isinstance(hla,list):
        if isinstance(hla,int):
            hla = [hla]
        else:
            return None, None
    # W = [np.random.randn(n,hla[0])]
    # b = [np.random.randn(hla[0])]
    W = [init((n,hla[0]))]
    b = [init([hla[0]])]
    for i in range(len(hla[:-1])):
        # W.append(np.random.randn(hla[i],hla[i+1]))
        # b.append(np.random.randn(hla[i+1]))
        W.append(init((hla[i],hla[i+1])))
        b.append(init([hla[i+1]]))
    W.append(init((hla[-1],r)))
    b.append(init([r]))
    return W, b
def forward(W,b,X):
    h = [X]
    for i in range(len(W)):
        h.append(sigmoid(h[-1]@W[i]+b[i]))
    return h

def add_pr(a,gr):
    gra = gr
    while np.ndim(gra) > len(a.shape):
        gra = np.sum(gra,axis=0)
    for ax,s in enumerate(a.shape):
        if s == 1:
            gra = np.sum(gra, axis= ax, keep_dims=True)
    return gra

def reduce_sum_pr(inp,gr,ax):
    sh = np.array(inp.shape)
    sh[ax] = 1
    t = inp.shape//sh
    gr = np.reshape(gr,sh)
    return np.tile(gr, t)

def backward(h,W,b,prev_grad=1):
    pg = [prev_grad*h[-1]*(1-h[-1])]
    W_grads = []
    b_grads = []
    for i in range(len(h)-1,0,-1):
        # dht/dht-1
        dht_1 = pg[-1]@W[i-1].T
        # dht/dwt-1
        dwt_1 = h[i-1].T@pg[-1]
        W_grads.insert(0, dwt_1)
        # dht/dbt-1
        dbt_1 = add_pr(b[i-1],pg[-1])
        b_grads.insert(0, dbt_1)

        pg.append(dht_1*h[i-1]*(1-h[i-1]))

    return W_grads, b_grads

def dJ_dh_last(y_hat,y):# y_hat = h[-1]
    gr = reduce_sum_pr(np.sum(y_hat-y,axis=1),1,ax=None)
    gra = reduce_sum_pr(y_hat-y,gr,ax=1)
    grad = 1/len(y)*gra*(y_hat-y)
    return grad

def J(y_hat, y):
    return 1./(2*len(y))*np.sum(np.square(y_hat-y))

def test(x_test, y_test, den):#, W_vals, b_vals):
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

def main():
    global W_vals, b_vals
    W, b = make(2,2,2)
    x_train = np.array([[1,1],[0,1],[1,0],[0,0]])
    y_train = np.array([[0,1],[1,0],[1,0],[0,1]])

    bs = 1
    dl = DataLoader(x_train, y_train, bs)
    dl.shuffle()

    # red_points = np.concatenate((
    #     0.2 * np.random.randn(25,2) + np.array([[0,0]]*25),
    #     0.2 * np.random.randn(25,2) + np.array([[1,1]]*25)
    #     ))
    # blue_points = np.concatenate((
    #     0.2 * np.random.randn(25,2) + np.array([[0,1]]*25),
    #     0.2 * np.random.randn(25,2) + np.array([[1,0]]*25)
    #     ))

    # y = [[1,0]]*len(blue_points) + [[0,1]] * len(red_points)
    s = 1
    converged = False
    while not converged:
        # h = forward(W,b,np.concatenate((blue_points, red_points)))
        for i, (x, y) in enumerate(dl.get_iterator()):
            h = forward(W,b,x)
            grad = dJ_dh_last(h[-1],y)
            J_val = J(h[-1],y)
            print(s,i,J_val)
            if s == 10:
                converged=True
            # if J_val < 0.12 or s > 2000:
            #     converged = True
            #     break
            W_grads, b_grads = backward(h,W,b,prev_grad=grad)
            for i in range(len(W)):
                W[i] -= 0.1 * W_grads[i]
                b[i] -= 0.1 * b_grads[i]
        s+=1

    # df = pd.read_csv('test.csv',header=None)
    # df = pd.read_csv('val.csv',header=None)
    # x_test = df.iloc[:,:-10].values
    # y_test = df.iloc[:,-10:].values
    x_test = []
    y_test = []
    for x, y in dl.get_iterator():
        for x_ in x:
            x_test.append(x_)
        for y_ in y:
            y_test.append(y_)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    den = len(x_test)

    W_vals = [w for w in W]
    b_vals = [b_ for b_ in b]
    end_ind = 0
    acc = 0
    while end_ind < den:
        s_ind = end_ind
        end_ind = min(s_ind+4000,den)
        x_test_slice = x_test[s_ind:end_ind]
        y_test_slice = y_test[s_ind:end_ind]
        acc += test(x_test_slice, y_test_slice,den)
    print(acc*100)
# def main():
#     global W_vals, b_vals
#     W, b = make(20,85,10)
#     df = pd.read_csv('train.csv',header=None)
#     x_train = df.iloc[:,:-10].values
#     y_train = df.iloc[:,-10:].values
# 
#     bs = 100
#     dl = DataLoader(x_train, y_train, bs)
#     dl.shuffle()
# 
#     # red_points = np.concatenate((
#     #     0.2 * np.random.randn(25,2) + np.array([[0,0]]*25),
#     #     0.2 * np.random.randn(25,2) + np.array([[1,1]]*25)
#     #     ))
#     # blue_points = np.concatenate((
#     #     0.2 * np.random.randn(25,2) + np.array([[0,1]]*25),
#     #     0.2 * np.random.randn(25,2) + np.array([[1,0]]*25)
#     #     ))
# 
#     # y = [[1,0]]*len(blue_points) + [[0,1]] * len(red_points)
#     for s in range(10):
#         # h = forward(W,b,np.concatenate((blue_points, red_points)))
#         for i, (x, y) in enumerate(dl.get_iterator()):
#             h = forward(W,b,x)
#             grad = dJ_dh_last(h[-1],y)
#             J_val = J(h[-1],y)
#             if not i % 100:
#                 print(s,i,J_val)
#             W_grads, b_grads = backward(h,W,b,prev_grad=grad)
#             for i in range(len(W)):
#                 W[i] -= 0.1 * W_grads[i]
#                 b[i] -= 0.1 * b_grads[i]
# 
#     # df = pd.read_csv('test.csv',header=None)
#     df = pd.read_csv('val.csv',header=None)
#     x_test = df.iloc[:,:-10].values
#     y_test = df.iloc[:,-10:].values
#     # x_test = [x for x, y in dl.get_iterator()]
#     # y_test = [y for x, y in dl.get_iterator()]
#     den = len(x_test)
# 
#     W_vals = [w for w in W]
#     b_vals = [b_ for b_ in b]
#     end_ind = 0
#     acc = 0
#     while end_ind < den:
#         s_ind = end_ind
#         end_ind = min(s_ind+4000,den)
#         x_test_slice = x_test[s_ind:end_ind]
#         y_test_slice = y_test[s_ind:end_ind]
#         acc += test(x_test_slice, y_test_slice,den)
#     print(acc*100)

main()
