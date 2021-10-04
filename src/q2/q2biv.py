import logging
import argparse
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from libsvm.svmutil import *

logging.basicConfig(level=logging.INFO)
matplotlib.rcParams['text.usetex'] = True

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-train",type=str,help='Path to training dataset',required=True)
    parser.add_argument("-test",type=str,help='Path to test dataset',required=True)
    args = parser.parse_args()
    return args

def getdata(path):
    df = pd.read_csv(path,header=None).astype(dtype=np.float)
    x_df = df.iloc[:,:-1]/255.0
    y_df = df.iloc[:,-1]
    return x_df, y_df

def getsplits(m,k):
    rp = np.random.permutation(m)
    K = np.split(rp, k)
    return K

def do_validation(x_df, y_df):
    # STEP0
    C_arr = [1e-5,1e-3,1.,5.,10.]
    # STEP1
    K = getsplits(x_df.shape[0], len(C_arr))
    # STEP2
    models = []
    acc = []
    best_model = None
    best_acc = -float('inf')
    best_c = None
    for idx, k in enumerate(K):
        logging.info(f"\n\tValidation: Setting C={C_arr[idx]}")
        x_train = x_df.iloc[x_df.index.difference(k),:].values.tolist()
        y_train = y_df.iloc[y_df.index.difference(k)].values.tolist()
        #
        x_val = x_df.iloc[k,:].values.tolist()
        y_val = y_df.iloc[k].values.tolist()
        #
        prob = svm_problem(y_train,x_train)
        param = svm_parameter(f'-t 2 -s 0 -g 0.05 -c {C_arr[idx]} -h 0')
        #
        m = svm_train(prob,param)
        models.append(m)
        #
        l,a,v = svm_predict(y_val, x_val, m)
        ACC, _, _ = evaluations(y_val, l)
        acc.append(ACC)
        if ACC > best_acc:
            best_model = m
            best_acc = ACC
            best_c = C_arr[idx]
    return {'models':models,'accuracies':acc,'best_model':best_model,'best_acc':best_acc,'best_c':best_c}

def do_tests(path,models):
    #
    x_df, y_df = getdata(path)
    x_test = x_df.values.tolist()
    y_test = y_df.values.tolist()
    #
    acc = []
    best_acc = -float('inf')
    best_c = None
    C_arr = [1e-5,1e-3,1.,5.,10.]
    for idx,m in enumerate(models):
        logging.info(f"\n\tTest: Model with C={C_arr[idx]}")
        l,a,v = svm_predict(y_test,x_test,m)
        ACC, _, _ = evaluations(y_test, l)
        acc.append(ACC)
        if ACC > best_acc:
            best_acc = ACC
            best_c = C_arr[idx]
    return {'accuracies':acc,'best_c':best_c}

def plot_graphs(val_acc,test_acc):
    C_arr = [1e-5, 1e-3, 1., 5., 10.]
    plt.plot(C_arr,val_acc,linestyle="--",marker='s',label='Validation Accuracy')
    plt.plot(C_arr,test_acc,linestyle=":",marker='^',label='Test Accuracy')
    plt.xscale('log')
    plt.xlabel('$\log C$')
    plt.ylabel('$Accuracy$')
    plt.legend()
    plt.savefig('kfold_cross_validation.png')
    plt.show()

def main():
    # STEP0
    args = getargs()
    x_df, y_df = getdata(args.train)
    C_arr = [1e-5,1e-3,1.,5.,10.]
    # STEP1
    results = do_validation(x_df, y_df)
    # STEP2
    test_results = do_tests(args.test,results['models'])
    # STEP3
    plot_graphs(results['accuracies'], test_results['accuracies'])
    # STEP4
    logging.info(f"\tValue of C with best accuracy for validation:{results['best_c']}")
    logging.info(f"\tValue of C with best accuracy for test dataset:{test_results['best_c']}")

if __name__ == "__main__":
    main()
