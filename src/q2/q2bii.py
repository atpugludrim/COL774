import time
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from libsvm.svmutil import *

from q2b_vectorized import getdata

def train(args):
    x_df,y_df = getdata(args.path_train)
    x_train = x_df.values.tolist()
    y_train = y_df.values.tolist()
    # x.shape should be (-1,784)

    prob = svm_problem(y_train,x_train)
    param = svm_parameter('-t 2 -s 0 -c 1 -g 0.05')
    m = svm_train(prob,param)
    svm_save_model(f'libsvm_multiclass.model',m)

def test(args):
    x_df,y_df = getdata(args.path_test)
    x_test = x_df.values.tolist()
    y_test = y_df.values.tolist()
    # x.shape should be (-1,784)

    m = svm_load_model(f'libsvm_multiclass.model')
    p_label, p_acc, p_val = svm_predict(y_test,x_test,m)
    ACC, MSE, SCC = evaluations(y_test, p_label)
    #with open('true_multiclass.csv','w') as f:
    #    for y in y_test:
    #        f.write(f"{y}\n")
    with open('pred_multiclass_libsvm.csv','w') as f:
        for l in p_label:
            f.write(f'{l}\n')
    print(ACC)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-train')
    parser.add_argument('--path-test',help="Training and testing are mutually exclusive. Do not define --path-train for testing.")
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
        print("Provide either a training path or testing path.\n\nusage: q2aiii.py [-h] [--path-train PATH_TRAIN] [--path-test PATH_TEST] [--quiet]\n")

if __name__=="__main__":
    s = time.perf_counter()
    main()
    e = time.perf_counter()
    logging.info("\tIt took {:.3f}s".format(e-s))
