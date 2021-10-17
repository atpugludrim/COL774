import sys
import time
import argparse
import numpy as np
import pandas as pd

sys.setrecursionlimit(3000)
class AttributeNotFound(Exception):
    pass

class DummyFile:
    def write(this,s):pass
    def flush(this,s):pass

class silentcontext:
    def __init__(this,verbose=False):
        this.verbose = verbose

    def __enter__(this):
        if this.verbose:
            return
        this.stdo = sys.stdout
        sys.stdout = DummyFile()

    def __exit__(this,exc_type,exc_value,traceback):
        if this.verbose:
            return
        sys.stdout = this.stdo

def silentdecorator(flag):
    def actualdecorator(func):
        def f(*args):
            if flag:
                stdo = sys.stdout
                sys.stdout = DummyFile()
            ret = func(*args)
            if flag:
                sys.stdout = stdo
            return ret
        return f
    return actualdecorator

def timeitdecorator(func):
    def f(*args):
        start = time.time()
        ret = func(*args)
        end = time.time()
        print("\nTime taken to run \"{}\" is {:.4f}s".format(func.__name__,(end-start)))
        return ret
    return f

class tree_node:
    def __init__(this,_type=None,attrib=None,disc_splits=None,cont_th=None,leaf_cl=None):
        this._type = _type
        this.attrib = attrib
        this.disc_splits = disc_splits# will be a dictionary
        this.cont_th = cont_th
        this.leaf_cl = leaf_cl
        this.children = []# will only have two children to be used for continuous splitting
    def add_child(this, child):# add left child first and then right
        this.children.append(child)
    def decide(this,x=None):
        if this._type == 'leaf':
            return this.leaf_cl
        elif this._type == 'disc':
            return this.disc_decide(x)
        elif this._type == 'cont':
            return this.cont_decide(x)
        else:
            return None
    def disc_decide(this,x):
        if this.attrib not in x.columns:
            raise AttributeNotFound()
        return this.disc_splits[x[this.attrib]]
    def cont_decide(this,x):
        if this.attrib not in x.columns:
            raise AttributeNotFound()
        if x[this.attrib] < this.cont_th:
            return this.children[0]
        else:
            return this.children[1]

def growtree(xdf,ydf):
    root = recursive_grow_tree(xdf,ydf,3)
    return root

def recursive_grow_tree(xdf,ydf,maxleaf):# MAX RECURSION DEPTH REACHED, MAKE THIS ITERATIVE
    # checkleafcondition()# and make leaf
    if len(np.unique(ydf)) == 1:
        print("\n","*"*30,"Leaf created","*"*30,"\n")
        return tree_node(_type = 'leaf', leaf_cl = ydf.values[0])
    if ydf.shape[0] <= maxleaf: # BECAUSE TWO OF ONE CLASS EACH WAS CAUSING INFINITE LOOP
        print("\n","*"*30,"Leaf created","*"*30,"\n")
        y_levels,count = np.unique(ydf,return_counts=True)
        return tree_node(_type='leaf',leaf_cl = y_levels[np.argmax(count)])

    #choose_best_attrib_to_split()
    MI = -float('inf')
    best_attrib = None
    best_attrib_th = None
    best_attrib_type = None
    y = ydf.values
    with silentcontext():
        for j in xdf.columns:
            if j == 'balance' or j == 'duration':# THIS HAS TO BE REMOVED
                continue
            xj = xdf[j].values
            if xj.dtype == np.dtype(np.int64):
                mi,th = mutual_information_j_continuous(y,xj)
                print(j,mi,th)
            else:
                mi = mutual_information_j(y,xj)
                print(j,mi)
            if mi > MI:
                MI = mi
                best_attrib = j
                if xj.dtype == np.dtype(np.int64):
                    best_attrib_th = th
                    best_attrib_type = 'cont'
                else:
                    best_attrib_type = 'disc'
    print("\nBest attrib to split on is",best_attrib)
    if best_attrib_type == 'cont':
        print("with threshold",best_attrib_th)

    #create_split()# and add child
    if best_attrib_type == 'disc':# make disc_splits dict
        disc_splits = dict()
        x_levels = np.unique(xdf[best_attrib])
        for l in x_levels:
            ind = xdf[best_attrib]==l
            disc_splits[l] = recursive_grow_tree(xdf[ind],ydf[ind],maxleaf)
        root = tree_node(_type=best_attrib_type,disc_splits=disc_splits,attrib=best_attrib)
    elif best_attrib_type == 'cont':# add left and right child
        root = tree_node(_type=best_attrib_type,cont_th=best_attrib_th,attrib=best_attrib)

        ind = xdf[best_attrib] < best_attrib_th
        root.add_child(recursive_grow_tree(xdf[ind],ydf[ind],maxleaf))#left child

        ind = xdf[best_attrib] > best_attrib_th
        root.add_child(recursive_grow_tree(xdf[ind],ydf[ind],maxleaf))#right child
    return root

def entropy(ps):
    if type(ps) != np.ndarray:
        ps = np.array(ps)
    H = -np.sum(ps*np.log(ps+1e-30))
    return H

def mutual_information_j(Y, Xj):
    x_levels, counts = np.unique(Xj, return_counts=True)

    if type(Y) != np.ndarray:
        Y = np.array(Y)
    if type(Xj) != np.ndarray:
        Xj = np.array(Xj)

    lx = len(Xj)
    p_x = [c/lx for c in counts]
    H = 0
    
    for i, x in enumerate(x_levels):
        # according to timeit, it's taking around 800 ms for
        # every feature
        y_cond = Y[np.where(Xj==x)]
        y_levels, counts = np.unique(y_cond, return_counts=True)
        ly = len(y_cond)

        ps = [c/ly for c in counts]
        H += p_x[i] * entropy(ps)

    y_levels, counts = np.unique(Y, return_counts=True)
    ly = len(Y)
    ps = [c/ly for c in counts]
    return entropy(ps) - H

def mutual_information_j_continuous(Y, Xj):# outermost calls are taking around 10 seconds in worst case, but on average is working under 2s
    x_sorted = np.unique(Xj)
    x_levels = []
    if len(x_sorted) == 1:
        x_sorted = np.append(x_sorted,x_sorted[0]+1)
    for i in range(len(x_sorted)-1):
        x_levels.append((x_sorted[i]+x_sorted[i+1])/2)

    if type(Y) != np.ndarray:
        Y = np.array(Y)
    if type(Xj) != np.ndarray:
        Xj = np.array(Xj)

    lx = len(Xj)
    H = float('inf')
    best_th = None
    
    for i, x in enumerate(x_levels):
        Hx = 0
        px = np.sum(Xj<x)/lx
        y_less = Y[np.where(Xj<x)]
        _, counts = np.unique(y_less,return_counts=True)
        ly = len(y_less)
        ps = [c/ly for c in counts]
        Hx += px * entropy(ps)

        px = np.sum(Xj>x)/lx
        y_great = Y[np.where(Xj>x)]
        _, counts = np.unique(y_great,return_counts=True)
        ly = len(y_great)
        ps = [c/ly for c in counts]
        ps = []
        Hx += px * entropy(ps)

        if Hx < H:
            H = Hx
            best_th = x

    ly = len(Y)
    y_levels, counts = np.unique(Y, return_counts=True)
    ps = [c/ly for c in counts]
    return entropy(ps) - H, best_th

@silentdecorator(True)
def test(root, xdf, ydf):
    ypred = []
    ytrue = []
    for i in range(len(xdf)):
        x = xdf.iloc[i,:]
        try:
            y = root.decide(x)
            while(type(y) == tree_node):
                y = y.decide(x)
            print(y,ydf.iloc[i])
            ypred.append(y)
            ytrue.append(ydf.iloc[i])
        except AttributeNotFound:
            print("Sample has unidentified attributes, cannot process",file=sys.stderr)
    return np.array(ypred), np.array(ytrue)

@timeitdecorator
def main():
    df = pd.read_csv('/home/anupam/Desktop/backups/COL774/data/q1/bank_train.csv',delimiter=';')
    #df = pd.read_csv('/home/anupam/Desktop/backups/COL774/src/q1/test.csv',delimiter=';')
    xdf = df.iloc[:,:-1]
    ydf = df.iloc[:,-1]

    root = growtree(xdf, ydf)

    df = pd.read_csv('/home/anupam/Desktop/backups/COL774/data/q1/bank_test.csv',delimiter=';')
    xdf = df.iloc[:,:-1]
    ydf = df.iloc[:,-1]
    y_hat,y = test(root,xdf,ydf)
    print(np.sum(y==y_hat)/y.shape[0])

if __name__ == "__main__":
    main()
