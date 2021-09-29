import re
import json
import time
import pickle
import string
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from q1a import parse, remove_punc

stop_words = set(stopwords.words('english'))
punkt = string.punctuation
ps = PorterStemmer()
def remove_stopwords(text):
    global stop_words
    splitstring = "\s|\.|,|\(|\)|-|\\\\|\"|!|#|\$|%|&|/|\*|\+|:|;|<|=|>|\?|@|\[|\]|\^|_|\`|\{|\}|\||~"
    words = [w.strip(punkt) for w in re.split(splitstring,text.strip())]
    new_words = []
    for w in words:
        if w.lower() not in stop_words:
            new_words.append(w)
    return " ".join(new_words)

def dostemming(text):
    global ps
    words = text.split()
    new_words = []
    for w in words:
        new_words.append(ps.stem(w))
    return " ".join(new_words)

def train(dpath,of):
    v = []
    _total = 0
    _phi = [0.0 for _ in range(5)]
    _nume = [dict() for _ in range(5)]
    _deno = [0.0 for _ in range(5)]
    #############################################################
    for i, l in enumerate(parse(dpath)):
        logging.warning("Parsing record: {}".format(i))
        k = int(float(l['overall']))-1 # ZERO INDEXING
        x = dostemming(remove_punc(remove_stopwords(l['reviewText']))).split()
        _total += 1
        _phi[k] += 1
        _deno[k] += len(x)
        updated = []
        for t in x:
            logging.debug("Word: {}".format(t))
            if t not in v:
                v.append(t)

            for idx in range(5):
                if t not in _nume[idx]:
                    _nume[idx][t] = 0.0
            if t in updated:
                continue
            updated.append(t)
            _nume[k][t] += np.sum(np.array(x)==t)
        #
    # CHECKING PROBABILITIES SUM TO 1 BEFORE SMOOTHING
    sums = [0 for _ in range(5)]
    for k in range(5):
        _phi[k] /= _total
        for word in _nume[k]:
            if _deno[k]:
                sums[k] += _nume[k][word]/_deno[k]
    logging.warning("Sums of theta for each class: {}".format(",".join(["{:.2f}".format(s) for s in sums])))
    # SMOOTHING
    thetas = [dict() for _ in range(5)]
    alpha = 1
    for k,n in enumerate(_nume):
        for word in n:
            logging.warning("smoothing: {} {}".format(k,word))
            thetas[k][word] = (_nume[k][word]+alpha)/(_deno[k]+alpha*len(v))
    logging.warning(len(v))
    logging.debug("phi:{}\nth:{}".format(_phi,thetas))
    with open('{}.pkl'.format(of),'wb') as f:
        pickle.dump({'phi':_phi,'thetas':thetas},f)

def test(dpath,thetas_file):
    with open("{}.pkl".format(thetas_file),'rb') as f:
        parameters = pickle.load(f)
    thetas = parameters['thetas']
    phi = parameters['phi']

    true = []
    pred = []
    total = 0
    for l in parse(dpath):
        total += 1
        k = int(float(l['overall']))-1 # ZERO INDEXING
        x = dostemming(remove_punc(remove_stopwords(l['reviewText']))).split()

        true.append(k+1)
        log_proba = [0.0 for _ in range(5)]
        for t in x:
            for k in range(5):
                logging.info("Word: {}".format(t))
                if t in thetas[k]:
                    _PHI = phi[k]
                    if _PHI == 0:
                        _PHI = 1e-30 # FOR STABILITY
                    log_proba[k] += np.log(thetas[k][t])+np.log(_PHI)
        pred.append(np.argmax(log_proba)+1)
        logging.warning("Parsing record: {}\tPrediction: {}\tTrue: {}".format(total,pred[-1],true[-1]))
    print("Accuracy:",(np.sum(np.array(true)==np.array(pred))/total*100),"%")
    logging.info("{}\n{}".format(true,pred))

    # FOR CONFUSION MATRIX
    with open('true.pkl','wb') as f:
        pickle.dump(true,f)
    with open('pred.pkl','wb') as f:
        pickle.dump(pred,f)

def main():
    parser = argparse.ArgumentParser(usage="use -h for list of options")
    parser.add_argument('-p',required=True,help='path to data')
    parser.add_argument('--logging-level',default="warning",type=str)
    parser.add_argument('--train',default=False,action='store_true')
    parser.add_argument('--output_file',default="theta_ls",type=str,help='theta dictionaries will be output to this file for training; .pkl will be appended automatically')
    parser.add_argument('--thetas_file',default="theta_ls",type=str,help='theta dictionaries will be loaded from this file for testing; .pkl will be appended automatically')
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging,args.logging_level.upper()),format='%(name)s - %(message)s\r')
    if args.train:
        train(args.p,args.output_file)
    else:
        test(args.p,args.thetas_file)

if __name__=="__main__":
    s = time.perf_counter()
    main()
    e = time.perf_counter()
    logging.warning("It took {:.3f}s".format(e-s))
    if (e-s) > 100:
        logging.warning("That's too long, this is not good.")
