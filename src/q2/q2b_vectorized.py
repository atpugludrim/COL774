import logging
import time
import pandas as pd
import numpy as np
import cvxopt
import pickle
import argparse
from scipy.spatial.distance import cdist

def getdata(path):
    df = pd.read_csv(path,header = None).astype(dtype=np.float)
    x_df = df.iloc[:,:-1]/255.0 # RETURNING SCALED DATA DIRECTLY
    y_df = df.iloc[:,-1]
    return x_df,y_df

def GaussianKernel(x,y):
    Y = y.reshape(-1,1)*1.
    Y_ = Y@Y.T
    P = calcgausskern(x)
    P = np.multiply(P,Y_)
    return P

def calcgausskern(x):
    gamma = 0.05
    P = np.exp(-gamma*np.square(cdist(x,x,'euclidean')))
    return P

def make_matrices(x,y,save_P,name):
    m = x.shape[0]
    assert m == y.shape[0]
    c = 1
    if save_P:
        P = GaussianKernel(x,y)
        P = 0.5*P
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

def train_util(x_train_normalized,y_train,savep,class1,class2):
    logging.info("\tgetting matrices now")

    P,q,G,h,A,b = make_matrices(x_train_normalized,y_train,savep,f"{class1}{class2}")
    logging.info("\tinvoking solver now")

    sol = cvxopt.solvers.qp(P,q,G,h,A,b)
    alphas = np.array(sol["x"])
    
    logging.info(f"\tCalculating w and b for {class1}(-1) vs {class2}(+1)")
    svind = np.logical_and(alphas > 1e-4, alphas <= 1).reshape((-1,))

    w = {'xi':x_train_normalized[svind],'yi':y_train[svind],'ai':alphas[svind]}
    b = np.sum(y_train[svind] - calcgausskern(x_train_normalized[svind]).T@np.multiply(y_train[svind],alphas[svind].reshape((-1))))/np.sum(svind)
    return w,b


def train(args):
    x_df,y_df = getdata(args.path_train)
    for class1 in range(10):
        for class2 in range(class1+1,10):
            # CREATE X AND Y
            ind = ((y_df == class1) | (y_df == class2))
            x_train = x_df[ind]
            y_train = y_df[ind]
            y_train[y_train == class1] = -1 # LOWER CLASS IS -1
            y_train[y_train == class2] = 1 # HIGHER CLASS IS +1

            w,b = train_util(x_train.values,y_train.values,args.savep,class1,class2)
            with open(f"wb_{class1}{class2}.pkl","wb") as f:
                pickle.dump({'w':w,'b':b},f)
                logging.info(f"\tw and b saved to wb_{class1}{class2}.pkl")
    logging.info("\tTraining complete")

def test(args):
    x_df,y_df = getdata(args.path_test)
    x_test_normalized = x_df.values
    y_test = y_df.values

    logging.info('\tData loaded')
    # x.shape should be (-1,784)
    ws = []
    bs = []
    for class1 in range(10):
        for class2 in range(class1+1,10):
            with open(f"wb_{class1}{class2}.pkl","rb") as f:
                wb = pickle.load(f)
                ws.append(wb['w'])
                bs.append(wb['b'])
    logging.info('\tParameters loaded')

    def sign(x):
        return -1 if x < 0 else 1

    logging.info('\tTesting')

    ctr = 0
    m = x_test_normalized.shape[0]
    gamma = 0.05
    ovo_y_hats = np.zeros((m,45))

    for class1 in range(10):
        for class2 in range(class1+1,10):
            logging.info(f"\tTesting {class1} vs {class2}")
            n = ws[ctr]['xi'].shape[0]
            K = np.exp(-gamma*np.square(cdist(ws[ctr]['xi'],x_test_normalized)))
            ay = np.multiply(ws[ctr]['yi'],ws[ctr]['ai'].reshape(-1)).reshape(-1,1)
            d = K.T@ay+np.repeat(bs[ctr],m).reshape(-1,1)
            # y_hat_ = np.where(d<0,-1,1)
            ovo_y_hats[np.where(d<0),class1] += 1
            ovo_y_hats[np.where(d>=0),class2] += 1
            ctr += 1
    y_hat = np.argmax(ovo_y_hats,axis=1)

    with open('true_multiclass.csv','w') as f:
        for y in y_test:
            f.write(f"{y}\n")
    with open('pred_multiclass.csv','w') as f:
        for y_ in y_hat.reshape(-1):
            f.write(f"{y_}\n")
    logging.info("\tStoring outputs in true.csv and pred_multiclass.csv")
    acc = np.sum(y_test == y_hat)*100.0/y_test.shape[0]
    logging.info("\tAccuracy is : {:.2f}%".format(acc))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-train')
    parser.add_argument('--path-test',help="Training and testing are mutually exclusive. Do not define --path-train for testing.")
    parser.add_argument('--savep',default=False,action='store_true')
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
