import numpy as np
from libsvm.svmutil import *
import logging
import time
import pickle
import argparse
from q2a import getdata, Scaler
def train(args):
    x_train,y_train = getdata(args.path_train,ylab=2)
    # x.shape should be (-1,784)
    train_scaler = Scaler()
    x_train_normalized = train_scaler.transform(x_train)
    logging.info("\tx_train normalized")

    prob = svm_problem(y_train,x_train_normalized)
    if args.kernel == "linear":
        param = svm_parameter('-t 0 -s 0 -c 1')
    else:
        param = svm_parameter('-t 2 -s 0 -c 1 -g 0.05')
    m = svm_train(prob,param)
    svm_save_model(f'libsvm_{args.kernel}.model',m)

def test(args):
    x_test,y_test = getdata(args.path_test,ylab=2)
    # x.shape should be (-1,784)
    test_scaler = Scaler()
    x_test_normalized = test_scaler.transform(x_test)
    logging.info("\tx_test normalized")

    m = svm_load_model(f'libsvm_{args.kernel}.model')
    p_label, p_acc, p_val = svm_predict(y_test.tolist(),x_test_normalized.tolist(),m)
    ACC, MSE, SCC = evaluations(y_test.tolist(), p_label)
    print(ACC)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-train')
    parser.add_argument('--path-test',help="Training and testing are mutually exclusive. Do not define --path-train for testing.")
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
        print("Provide either a training path or testing path.\n\nusage: q2aiii.py [-h] [--path-train PATH_TRAIN] [--path-test PATH_TEST] --kernel {gaussian,linear} [--quiet]\n")

if __name__=="__main__":
    s = time.perf_counter()
    main()
    e = time.perf_counter()
    logging.info("\tIt took {:.3f}s".format(e-s))
