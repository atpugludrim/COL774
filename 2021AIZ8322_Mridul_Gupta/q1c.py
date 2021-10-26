import sys
import time
import copy
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC

class DummyFile:
    def write(this, *args, **kwargs):pass
    def flush(this, *args, **kwargs):pass

sys.stdout = DummyFile() # silencing all outputs
sys.stderr = DummyFile() # silencing all error outputs
def timeitdec(func):
    def f(*args):
        s = time.time()
        ret = func(*args)
        e = time.time()
        print("Aisa hai beta ki tumhe \"{}\" chalane mei {} seconds lage hai".format(func.__name__,(e-s)))
        return ret
    return f

def silentdec(silent=True):
    def f(func):
        def f1(*args):
            std = sys.stdout
            ste = sys.stderr
            print("Silencing \"{}\"".format(func.__name__),file=sys.stderr)
            sys.stdout = DummyFile()
            sys.stderr = DummyFile()
            ret = func(*args)
            sys.stdout = std
            sys.stderr = ste
            return ret
        if silent:
            return f1
        else:
            return func
    return f

def increase_like_clockwork(ind, max_ind, idx):
    if idx >= len(ind):
        return False
    ind[idx] += 1
    ret=True
    if ind[idx] >= max_ind[idx]:
        ind[idx] = 0
        ret=increase_like_clockwork(ind, max_ind, idx+1)
    return ret

@timeitdec
@silentdec(not True)
def main():
    trainp = sys.argv[1]
    valp = sys.argv[2]
    testp = sys.argv[3]
    #############################################################
    # 1) HAVE TO MAKE DUMMIES
    df = pd.read_csv(trainp,delimiter=';')
    xdf_train = df.iloc[:,:-1]
    xdf_train_oneh = pd.get_dummies(xdf_train)
    ts = len(df)#3000
    ind = np.random.permutation(len(xdf_train_oneh))[:ts]
    xdf_train_oh = xdf_train_oneh.iloc[ind]
    ydf_train = df.iloc[ind,-1]
    print("Training data loaded")

    # 2)
    df = pd.read_csv(testp,delimiter=';')
    xdf_test = df.iloc[:,:-1]
    xdf_test_oh = pd.get_dummies(xdf_test)
    missing_cols = set(xdf_train_oh.columns) - set(xdf_test_oh.columns)
    for c in missing_cols:
        xdf_test_oh[c] = 0
    xdf_test_oh = xdf_test_oh[xdf_train_oh.columns]
    ydf_test = df.iloc[:,-1]
    print("Testing data loaded")

    # 3)
    df = pd.read_csv(valp,delimiter=';')
    xdf_val = df.iloc[:,:-1]
    xdf_val_oh = pd.get_dummies(xdf_val)
    missing_cols = set(xdf_train_oh.columns) - set(xdf_val_oh.columns)
    for c in missing_cols:
        xdf_val_oh[c] = 0
    xdf_val_oh = xdf_val_oh[xdf_train_oh.columns]
    ydf_val = df.iloc[:,-1]
    print("Validation data loaded")

    # make param_grid
    param_grid = [OrderedDict({'n_estimators':[_ for _ in range(50,451,100)],
        'max_features':np.arange(0.1,1,0.2),
        'min_samples_split':np.arange(2,11,2)})]

    # MAKE CLASSIFIER WITH VARIOUS PARAMETERS
    best_model = None
    best_oob = -float('inf')
    best_params = {'n_estimators':None,'max_features':None,'min_samples_split':None}
    clf = RFC(oob_score=True)
    ind = [0 for _ in range(len(param_grid[0]))] # ind for each parameter
    max_ind = [len(param_grid[0][p]) for p in param_grid[0]] # max ind for each parameter

    ctr = 0
    while True:
        # code here
        ctr += 1
        print(f"training classifier {ctr}")
        for idx in range(len(ind)):
            setattr(clf,list(param_grid[0].keys())[idx],param_grid[0][list(param_grid[0].keys())[idx]][ind[idx]])
        clf.fit(xdf_train_oh.values, ydf_train.values)
        if clf.oob_score_ > best_oob:
            best_oob = clf.oob_score_
            best_params['n_estimators'] = param_grid[0]['n_estimators'][ind[idx]]
            best_params['max_features'] = param_grid[0]['max_features'][ind[idx]]
            best_params['min_samples_split'] = param_grid[0]['min_samples_split'][ind[idx]]
            best_model = copy.deepcopy(clf)
        if not increase_like_clockwork(ind, max_ind, 0):
            break

    with open("best_model.pkl".format(ts),"wb") as f:
        pickle.dump(best_model, f)
    with open("best_params.pkl".format(ts),"wb") as f:
        pickle.dump(best_params, f)

    y_hat_train = clf.predict(xdf_train_oh.values)
    y_hat_test = clf.predict(xdf_test_oh.values)
    y_hat_val = clf.predict(xdf_val_oh.values)
    
    acc_train = np.sum(y_hat_train == ydf_train.values)/y_hat_train.shape[0]
    acc_test = np.sum(y_hat_test == ydf_test.values)/y_hat_test.shape[0]
    acc_val = np.sum(y_hat_val == ydf_val.values)/y_hat_val.shape[0]

    f = open("accuracies_rf_{}.txt".format(ts),"w")
    f.write("Train {}\n".format(acc_train))
    f.write("Test {}\n".format(acc_test))
    f.write("Val {}\n".format(acc_val))
    f.close()

if __name__ == "__main__":
    main()
