import os
import time
import pickle
import logging
import argparse
import numpy as np
import pandas as pd

from scipy.sparse import lil_matrix as matrix

from q1d import train, test

def train_1(px,py,sx,of,th):
    if not os.path.isfile(th):# CHECKS IF 1D ALREADY RUN
        train(px,py,th[:-4])
    ys = pd.read_csv(py,header=None).astype(dtype=np.int).values.reshape(-1)
    _total = len(ys)
    _phi = [(1./_total)*np.sum(ys==(k+1)) for k in range(5)]
    vocab = pd.read_csv('vocab_summary.txt',header=None).astype(dtype=str).values.reshape(-1)
    m = len(vocab)
    _nume = matrix((m,5))
    _deno = [0.0 for _ in range(5)]
    #############################################################
    with open(sx,'r') as xf:
        for i, l in enumerate(xf):
            logging.info("Reading record: {}".format(i))
            x = l.split()
            _deno[ys[i]-1] += len(x)
            updated = []
            for t in x:
                # logging.info("Word: {}".format(t))
                if t in updated:
                    continue
                updated.append(t)
                try:
                    idx = np.searchsorted(vocab,t) # BINARY SEARCH
                except Exception as e:
                    logging.info(f"\tException{e}")
                    continue
                _nume[idx,ys[i]-1] += np.sum(np.array(x)==t)
        #
    # CHECKING PROBABILITIES SUM TO 1 BEFORE SMOOTHING
    # sums = [0 for _ in range(5)]
    # for k in range(5):
    #     _phi[k] /= _total
    #     for word in _nume[k]:
    #         if _deno[k]:
    #             sums[k] += _nume[k][word]/_deno[k]
    # logging.info("Sums of theta for each class: {}".format(",".join(["{:.2f}".format(s) for s in sums])))
    # SMOOTHING
    thetas = matrix((m,5))
    alpha = 1
    for j in range(5):
        for i in range(m):
            thetas[i,j] = np.log((_nume[i,j]+alpha)/(_deno[j]+alpha*m))
    for j in range(5):
        sum_ = 0
        for i in range(m):
            sum_ += np.exp(thetas[i,j])
        print("Theta sum for class ",j,sum_)
    with open('{}.pkl'.format(of),'wb') as f:
        pickle.dump({'phi':_phi,'thetas':thetas},f)

def test(px,py,thetas_file):
    with open("{}.pkl".format(thetas_file),'rb') as f:
        parameters = pickle.load(f)
    thetas = parameters['thetas']
    phi = parameters['phi']

    pred = []
    ys = pd.read_csv(py,header=None).astype(dtype=np.int).values.reshape(-1)
    total = len(ys)
    vocab = pd.read_csv('vocab_stem_stop.txt',header=None).astype(dtype=str).values.reshape(-1)
    with open(px,'r') as xf:
        for l in xf:
            x = l.split()

            log_proba = [0.0 for _ in range(5)]
            for t in x:
                try:
                    idx = np.searchsorted(vocab,t)
                except Exception as e:
                    logging.info(f"\tException {e}")
                    print(type(vocab[0]))
                    continue
                if vocab[idx] != t:
                    continue
                for k in range(5):
                    log_proba[k] += thetas[idx,k]
            for k in range(5):
                _PHI = phi[k]
                if _PHI == 0:
                    _PHI = 1e-30
                log_proba[k] += np.log(_PHI)
            lp = np.array(log_proba)-np.max(log_proba)
            s = np.sum(np.exp(lp))
            proba = []
            for k in range(5):
                proba.append(np.exp(lp[k])/s)
            pred.append(proba)
    return pred

def test_1(px,py,sx,thetas_file,th):
    pred_d = test(px,py,th[:-4])
    with open("{}.pkl".format(thetas_file),'rb') as f:
        parameters = pickle.load(f)
    thetas = parameters['thetas']
    phi = parameters['phi']

    pred = []
    ys = pd.read_csv(py,header=None).astype(dtype=np.int).values.reshape(-1)
    total = len(ys)
    vocab = pd.read_csv('vocab_summary.txt',header=None).astype(dtype=str).values.reshape(-1)
    with open(sx,'r') as xf:
        for idx, l in enumerate(xf):
            x = l.split()

            log_proba = [0.0 for k in range(5)]
            for t in x:
                try:
                    idx = np.searchsorted(vocab,t)
                except Exception as e:
                    logging.info(f"\tException {e}")
                    print(type(vocab[0]))
                    continue
                if vocab[idx] != t:
                    continue
                for k in range(5):
                    log_proba[k] += thetas[idx,k]
            for k in range(5):
                _PHI = phi[k]
                if _PHI == 0:
                    _PHI = 1e-30
                log_proba[k] += np.log(_PHI)
            lp = np.array(log_proba)-np.max(log_proba)
            s = np.sum(np.exp(lp))
            proba = []
            for k in range(5):
                proba.append(np.exp(lp[k])/s)
            if max(pred_d[idx]) > max(proba):
                pred.append(np.argmax(pred_d[idx])+1)
            else:
                pred.append(np.argmax(log_proba)+1)
    print("Accuracy:",(np.sum(ys==np.array(pred))/total*100),"%")
    #logging.info("{}\n{}".format(true,pred))

    # FOR CONFUSION MATRIX
    with open('true.pkl','wb') as f:
        pickle.dump(ys,f)
    with open('pred_1g.pkl','wb') as f:
        pickle.dump(pred,f)

def main():
    parser = argparse.ArgumentParser(usage="use -h for list of options")
    parser.add_argument('-px',required=True,help='path to data')
    parser.add_argument('-py',required=True,help='path to data')
    parser.add_argument('-sx',required=True,help='path to data')
    parser.add_argument('--logging-level',default="warning",type=str)
    parser.add_argument('--train',default=False,action='store_true')
    parser.add_argument('--output_file',default="theta_ls",type=str,help='theta dictionaries will be output to this file for training; .pkl will be appended automatically')
    parser.add_argument('--thetas_file',default="theta_ls",type=str,help='theta dictionaries will be loaded from this file for testing; .pkl will be appended automatically')
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging,args.logging_level.upper()),format='%(name)s - %(message)s\r')
    if args.train:
        if args.thetas_file is None:
            th = 'theta1d.pkl'
        else:
            th = f'{args.thetas_file}.pkl'
        train_1(args.px,args.py,args.sx,args.output_file,th)
    else:
        if os.path.isfile('theta_1d.pkl'):
            th = 'theta_1d.pkl'
        elif os.path.isfile('theta1d.pkl'):
            th = 'theta1d.pkl'
        test_1(args.px,args.py,args.sx,args.thetas_file,th)

if __name__=="__main__":
    s = time.perf_counter()
    main()
    e = time.perf_counter()
    logging.info("It took {:.3f}s".format(e-s))
    if (e-s) > 100:
        logging.info("That's too long, this is not good.")
