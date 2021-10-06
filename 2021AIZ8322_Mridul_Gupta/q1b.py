import time
import logging
import argparse
import numpy as np
import pandas as pd
from scipy import stats

def train(py,py_train):
    ys = pd.read_csv(py_train,header=None).astype(dtype=np.int).values.reshape(-1)
    ys_test = pd.read_csv(py,header=None).astype(dtype=np.int).values.reshape(-1)
    m = stats.mode(ys).mode.item()
    pred = np.random.choice(np.arange(1,6),size=ys_test.shape[0])
    print("Random accuracy is: {:.2f}%".format(np.sum(ys_test==pred)*100/ys_test.shape[0]))
    print("Maximum accuracy is: {:.2f}%".format(np.sum(ys_test==m)*100/ys_test.shape[0]))

def main():
    parser = argparse.ArgumentParser(usage="use -h for list of options")
    parser.add_argument('-py_train',required=True,help='path to data')
    parser.add_argument('-py',required=True,help='path to data')
    parser.add_argument('--logging-level',default="warning",type=str)
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging,args.logging_level.upper()),format='%(name)s - %(message)s\r')
    train(args.py,args.py_train)

if __name__=="__main__":
    s = time.perf_counter()
    main()
    e = time.perf_counter()
    logging.info("It took {:.3f}s".format(e-s))
    if (e-s) > 100:
        logging.info("That's too long, this is not good.")
