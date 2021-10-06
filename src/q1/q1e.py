import time
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix as matrix

vocab = None

def get_n_grams(X,n):
    global vocab
    x = np.array(X)
    indexer = np.arange(n)[None,:]+n*np.arange(len(X)//n)[:,None]
    ngrams = x[indexer]
    #ngrams_list = ['_'.join(n) for n in ngrams]
    ngrams_list = []
    for ng in ngrams:
        rep = ""
        for word in ng:
            try:
                idx = np.searchsorted(vocab,word)
            except Exception as e:
                logging.info(f"\tException {e}")
                continue
            if vocab[idx] != word:
                continue
            rep += "{}_".format(idx)
        ngrams_list.append(rep)
    return ngrams_list

def get_len(l):
    if l < 60:
        return "ZERO"
    elif l < 100:
        return "ONE"
    elif l < 250:
        return "TWO"
    else:
        return "THREE"

def train(px,py,of):
    ys = pd.read_csv(py,header=None).astype(dtype=np.int).values.reshape(-1)
    _total = len(ys)
    _phi = [(1./_total)*np.sum(ys==(k+1)) for k in range(5)]
    _nume = [dict() for _ in range(5)]
    _deno = [0.0 for _ in range(5)]
    ng_vocab = []
    #############################################################
    with open(px,'r') as xf:
        for i, l in enumerate(xf):
            logging.info("Parsing record: {}".format(i))
            k = ys[i]-1
            stemmed = l.split()
            if len(stemmed) > 400:
                continue
            x = get_n_grams(stemmed, 5)
            x.append(get_len(len(stemmed)))
            _deno[k] += len(x)
            for t in x:
                if t not in ng_vocab:
                    ng_vocab.append(t)
                for idx in range(5):
                    if t not in _nume[idx]:
                        _nume[idx][t] = 0.0
                _nume[k][t] += 1
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
    thetas = [dict() for _ in range(5)]
    alpha = 1
    for k,n in enumerate(_nume):
        for word in n:
            logging.info("smoothing: {} {}".format(k,word))
            thetas[k][word] = np.log((_nume[k][word]+alpha)/(_deno[k]+alpha*len(ng_vocab)))
    logging.info(len(ng_vocab))
    #logging.info("phi:{}\nth:{}".format(_phi,thetas))
    with open('{}.pkl'.format(of),'wb') as f:
        pickle.dump({'phi':_phi,'thetas':thetas},f)

def test(px,py,thetas_file):
    global vocab
    with open("{}.pkl".format(thetas_file),'rb') as f:
        parameters = pickle.load(f)
    thetas = parameters['thetas']
    phi = parameters['phi']

    ys = pd.read_csv(py,header=None).astype(dtype=np.int).values.reshape(-1)
    total = len(ys)
    pred = []
    with open(px,'r') as xf:
        for l in xf:
            total += 1
            stemmed = l.split()
            x = get_n_grams(stemmed, 5)
            x.append(get_len(len(stemmed)))

            log_proba = [0.0 for _ in range(5)]
            for t in x:
                for k in range(5):
                    #logging.info("Word: {}".format(t))
                    if t in thetas[k]:
                        log_proba[k] += thetas[k][t]
            for k in range(5):
                _PHI = phi[k]
                if _PHI == 0:
                    _PHI = 1e-30
                log_proba[k] += np.log(_PHI)
            pred.append(np.argmax(log_proba)+1)
            #logging.info("Parsing record: {}\tPrediction: {}\tTrue: {}".format(total,pred[-1],true[-1]))
    print("Accuracy:",(np.sum(ys==np.array(pred))/total*100),"%")
    #logging.info("{}\n{}".format(true,pred))

    # FOR CONFUSION MATRIX
    with open('true.pkl','wb') as f:
        pickle.dump(ys,f)
    with open('pred_1e.pkl','wb') as f:
        pickle.dump(pred,f)

def main():
    parser = argparse.ArgumentParser(usage="use -h for list of options")
    parser.add_argument('-px',required=True,help='path to data')
    parser.add_argument('-py',required=True,help='path to data')
    parser.add_argument('--logging-level',default="warning",type=str)# CHANGE DEFAULT TO WARNING
    parser.add_argument('--train',default=False,action='store_true')
    parser.add_argument('--output_file',default="theta_ls",type=str,help='theta dictionaries will be output to this file for training; .pkl will be appended automatically')
    parser.add_argument('--thetas_file',default="theta_ls",type=str,help='theta dictionaries will be loaded from this file for testing; .pkl will be appended automatically')
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging,args.logging_level.upper()),format='%(name)s - %(message)s\r')

    global vocab
    vocab = pd.read_csv('vocab_stem_stop.txt',header=None).astype(dtype=str).values.reshape(-1)
    if args.train:
        train(args.px,args.py,args.output_file)
    else:
        test(args.px,args.py,args.thetas_file)

if __name__=="__main__":
    s = time.perf_counter()
    main()
    e = time.perf_counter()
    logging.info("It took {:.3f}s".format(e-s))
    if (e-s) > 100:
        logging.info("That's too long, this is not good.")
