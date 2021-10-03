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
    def __init__(this, mu=0., sig=255.):
        this.mu = mu
        this.sig = sig
    def transform(this, data):
        return (data - this.mu)/this.sig
    def inv_transform(this, data):
        return (data * this.sig) + this.mu

def GaussianKernel(x,y):
    #gamma = 0.05
    #m = x.shape[0]
    #X = x[...,np.newaxis]
    #Z = np.repeat(X,m,axis=2)
    #X = Z.T
    #Y = y.reshape(-1,1)*1.
    #Y_ = Y@Y.T
    #P = np.multiply(np.exp(-gamma*np.sum(np.square(X-Z))),Y_)
    #return P
    #P = np.zeros((m,m))
    Y = y.reshape(-1,1)*1.
    Y_ = Y@Y.T
    #for j in range(m):
    #    z = x[j,:]
    #    col_j = np.exp(-gamma*np.sum(np.square(x - z),axis=1))
    #    P[:,j] = col_j
    P = calcgausskern(x)
    P = np.multiply(P,Y_)
    return P

    #t1 = np.sum(np.square(x-y))
    #t2 = -gamma # MINUS SIGN TAKEN CARE OF HERE
    #return np.exp(t1*t2)

def calcgausskern(xm):
    gamma = 0.05
    m = xm.shape[0]
    P = np.zeros((m,m))
    for j in range(m):
        xs = xm[j,:]
        P[:,j] = np.exp(-gamma*np.sum(np.square(xm-xs),axis=1))
    return P

def LinearKernel(x,y):
    Y = y.reshape(-1,1)*1.
    Xy = np.multiply(x,Y)
    return Xy@Xy.T

    #return np.sum(np.multiply(x,z))

def make_matrices(x,y,save_P,kernel,name):
    m = x.shape[0]
    assert m == y.shape[0]
    c = 1
    if save_P:
        if name.lower() == "linear" or name.lower() == "gaussian":
            P = kernel(x,y)
            P = 0.5*P
        #else:
        #    P = np.zeros((m,m))
        #    for i,x_ in enumerate(x):
        #        for j,z_ in enumerate(x):
        #            logging.info(f"\t{i}, {j}")
        #            P[i,j] = y[i]*y[j]*kernel(x_,z_)*0.5
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

def train(args):
    if args.kernel == "linear":
        kernel = LinearKernel
    else:
        kernel = GaussianKernel
    x_train,y_train = getdata(args.path_train,ylab=2)
    # x.shape should be (-1,784)
    train_scaler = Scaler()
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
    
    logging.info("\tCalculating w and b")
    svind = np.logical_and(alphas > 1e-4, alphas <= 1).reshape((-1,))

    if args.kernel == "linear":
        w = x_train_normalized[svind].T@np.multiply(y_train[svind],alphas[svind].reshape((-1,)))
        b = np.sum(y_train[svind] - x_train_normalized[svind]@w,axis=0)/np.sum(svind)
    else:
        w = {'xi':x_train_normalized[svind],'yi':y_train[svind],'ai':alphas[svind]}
        b = np.sum(y_train[svind] - calcgausskern(x_train_normalized[svind]).T@np.multiply(y_train[svind],alphas[svind].reshape((-1))))/np.sum(svind)

    with open(f"wb_{args.kernel}.pkl","wb") as f:
        pickle.dump({'w':w,'b':b},f)
        logging.info(f"\tw and b saved to wb_{args.kernel}.pkl")

def test(args):
    train_scaler = Scaler()
    x_test,y_test = getdata(args.path_test,ylab=2)
    logging.info('\tData loaded')
    # x.shape should be (-1,784)
    x_test_normalized = train_scaler.transform(x_test)
    logging.info('\tData normalized')
    with open(f"wb_{args.kernel}.pkl","rb") as f:
        wb = pickle.load(f)
    logging.info('\tParameters loaded')
    w = wb['w']
    b = wb['b']

    def sign(x):
        return -1 if x < 0 else 1

    logging.info('\tTesting')
    if args.kernel == 'linear':
        d = x_test_normalized@w.reshape(-1,1)+np.repeat(b,x_test_normalized.shape[0]).reshape(-1,1)
        y_hat = np.where(d<0,-1,1)
    else:
        m = x_test_normalized.shape[0]
        n = w['xi'].shape[0]
        K = np.zeros((n,m))
        gamma = 0.05
        for j in range(m):
            z = x_test_normalized[j]
            K[:,j] = np.exp(-gamma*np.sum(np.square(w['xi']-z),axis=1))
        ay = np.multiply(w['yi'],w['ai'].reshape(-1)).reshape(-1,1)
        d = K.T@ay+np.repeat(b,m).reshape(-1,1)
        y_hat = np.where(d<0,-1,1)

    with open('true.csv','w') as f:
        for y in y_test:
            f.write(f"{y}\n")
    with open(f'pred_{args.kernel}.csv','w') as f:
        for y_ in y_hat.reshape(-1):
            f.write(f"{y_}\n")
    logging.info(f"\tStoring outputs in true.csv and pred_{args.kernel}.csv")
    acc = np.sum(y_test == y_hat.reshape(-1))*100.0/y_test.shape[0]
    logging.info("\tAccuracy is : {:.2f}%".format(acc))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-train')
    parser.add_argument('--path-test',help="Training and testing are mutually exclusive. Do not define --path-train for testing.")
    parser.add_argument('--savep',default=False,action='store_true')
    parser.add_argument('--kernel',required=True,choices=["gaussian","linear"])
    parser.add_argument('--quiet',default=False,action='store_true')
    args = parser.parse_args()
    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    #############################################################

    if args.path_train is not None:
        train(args)
    elif args.path_test is not None:
        test(args)
    else:
        print("Provide either a training path or testing path.\n\nusage: q2a.py [-h] [--path-train PATH_TRAIN] [--path-test PATH_TEST] [--savep] --kernel {gaussian,linear} [--quiet]\n")

if __name__=="__main__":
    s = time.perf_counter()
    main()
    e = time.perf_counter()
    logging.info("\tIt took {:.3f}s".format(e-s))
