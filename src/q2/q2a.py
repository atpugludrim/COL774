import logging
import time
import numpy as np
import cvxopt
import pickle
import argparse

def getdata(path,ylab=-1):
    data_x = []
    data_y = []
    with open(path,"r") as f:
        for l in f:
            features = l.split(',')
            x = []
            for F in features[:-1]:
                x.append(int(F.strip()))
            if ylab == -1:
                y = int(features[-1].strip())
            else:
                y = -1 if int(features[-1].strip())!=ylab else 1
            data_x.append(x)
            data_y.append(y)
    return np.array(data_x, dtype=np.float), np.array(data_y)

class Scaler:
    def __init__(this, mu=0, sig=255):
        this.mu = mu
        this.sig = sig
        this.sig[np.where(sig==0)]=1
    def transform(this, data):
        return (data - this.mu)/this.sig
    def inv_transform(this, data):
        return (data * this.sig) + this.mu

def GaussianKernel(x,z):
    gamma = 0.05
    t1 = np.sum(np.square(x-z))
    t2 = -gamma # MINUS SIGN TAKEN CARE OF HERE
    return np.exp(t1*t2)

def LinearKernel(x,z):
    return np.sum(np.multiply(x,z))

def make_matrices(x,y,save_P,kernel,name):
    m = x.shape[0]
    assert m == y.shape[0]
    c = 1
    if save_P:
        P = np.zeros((m,m))
        for i,x_ in enumerate(x):
            for j,z_ in enumerate(x):
                logging.info(f"\t{i}, {j}")
                P[i,j] = y[i]*y[j]*kernel(x_,z_)*0.5
        with open(f"P_{name}.pkl","wb") as f:
            pickle.dump(P,f)
    else:
        with open(f"P_{name}.pkl","rb") as f:
            P = pickle.load(f)
    q = -np.ones((m,))
    G = np.concatenate([-np.eye(m),np.eye(m)])
    h = np.concatenate([np.zeros(m),np.ones(m)*c])
    A = y.reshape((1,m))
    b = np.zeros(1)
    return cvxopt.matrix(P,tc='d'),cvxopt.matrix(q,tc='d'),cvxopt.matrix(G,tc='d'),cvxopt.matrix(h,tc='d'),cvxopt.matrix(A,tc='d'),cvxopt.matrix(b,tc='d')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-train')
    parser.add_argument('--path-test')
    parser.add_argument('--savep',default=False,action='store_true')
    parser.add_argument('--kernel',required=True,choices=["gaussian","linear"])
    parser.add_argument('--quiet',default=False,action='store_true')
    args = parser.parse_args()
    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    #############################################################
    if args.kernel == "linear":
        kernel = LinearKernel
    else:
        kernel = GaussianKernel

    x_train,y_train = getdata(args.path_train,ylab=2)
    # x.shape should be (-1,784)
    train_scaler = Scaler(x_train.mean(axis=0),x_train.std(axis=0))
    x_train_normalized = train_scaler.transform(x_train)
    logging.info("\tx_train normalized")
    logging.info("\tgetting matrices now")

    P,q,G,h,A,b = make_matrices(x_train_normalized,y_train,args.savep,kernel,args.kernel)
    logging.info("\tinvoking solver now")

    sol = cvxopt.solvers.qp(P,q,G,h,A,b)
    logging.info("\tsolved, saving object to [sol.pkl]")

    with open("sol.pkl","wb") as f:
        pickle.dump(sol,f)
    alphas = np.array(sol["x"])
    with open("alphas.txt","w") as f:
        lines = []
        for a in alphas:
            lines.append("{}\n".format(a[0]))
        f.writelines(lines)
        logging.info("\tAlphas saved in alphas.txt")

    #x_test,y_test = getdata(args.path_test,ylab=2)
    ## x.shape should be (-1,784)
    #x_test_normalized = train_scaler.transform(x_test)

if __name__=="__main__":
    s = time.perf_counter()
    main()
    e = time.perf_counter()
    logging.info("It took {:.3f}s".format(e-s))
