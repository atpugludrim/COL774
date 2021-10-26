import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

sys.setrecursionlimit(3000)
class AttributeNotFound(Exception):
    pass

class DummyFile:
    def write(this,s):pass
    def flush(this,*args):pass

class silentcontext:
    def __init__(this,silent):
        this.verbose = not silent

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

classes = ['yes','no']
class tree_node:
    def __len__(this):
        return 0

    def __init__(this,_type=None,attrib=None,disc_splits=None,cont_th=None,leaf_cl=None,_empty_leaf=False,num_samples=0,depth=0):
        this._type = _type
        this.depth=depth
        this.attrib = attrib
        this.disc_splits = disc_splits# will be a dictionary
        this.cont_th = cont_th
        this.leaf_cl = leaf_cl
        this.num_samples = num_samples
        this._empty_leaf = _empty_leaf
        this.children = []# will only have two children to be used for continuous splitting
    def add_child(this, child):# add left child first and then right
        this.children.append(child)
    def decide(this,x=None,test2=False):
        if this._type == 'leaf':
            if this._empty_leaf:
                global classes
                return np.random.choice(classes)
            return this.leaf_cl
        elif this._type == 'disc':
            return this.disc_decide(x,test2)
        elif this._type == 'cont':
            return this.cont_decide(x)
        else:
            return None
    def disc_decide(this,x,test2):
        if x[this.attrib] not in this.disc_splits:
            if test2:
                return [this.disc_splits[k] for k in this.disc_splits.keys()] # when test2 is used
            return this.disc_splits[np.random.choice(list(this.disc_splits.keys()))]
            # raise AttributeNotFound()
        return this.disc_splits[x[this.attrib]]
    def cont_decide(this,x):
        if x[this.attrib] < this.cont_th:
            return this.children[0]
        else:
            return this.children[1]

def growtree(xdf,ydf):
    root = recursive_grow_tree(xdf,ydf,21,0)
    global counter, empty
    print("Number of nodes:",counter,"and empty leaves:",empty)
    return root

counter = 0
empty = 0
def recursive_grow_tree(xdf,ydf,maxleaf,depth):# MAX RECURSION DEPTH REACHED, MAKE THIS ITERATIVE
    # checkleafcondition()# and make leaf
    global counter
    counter += 1
    print("Node:",counter)
    if len(np.unique(ydf)) == 1:
        print("\n","*"*30,"Leaf created","*"*30,"\n")
        return tree_node(_type = 'leaf', leaf_cl = ydf.values[0], num_samples = len(ydf),depth=depth)

    if ydf.shape[0] <= maxleaf: # BECAUSE TWO OF ONE CLASS EACH WAS CAUSING INFINITE LOOP
        print("\n","*"*30,"Leaf created","*"*30,"\n")
        y_levels,count = np.unique(ydf,return_counts=True)
        if len(y_levels) == 0:
            empty += 1
            print("empty leaf",empty)
            return tree_node(_type='leaf',_empty_leaf=True,num_samples = 0,depth=depth)
        return tree_node(_type='leaf',leaf_cl = y_levels[np.argmax(count)],num_samples = len(ydf),depth=depth)

    #choose_best_attrib_to_split()
    MI = -float('inf')
    best_attrib = None
    best_attrib_th = None
    best_attrib_type = None
    y = ydf.values
    with silentcontext(True):
        for j in xdf.columns:
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
    y_levels,counts = np.unique(ydf, return_counts=True)
    max_c = y_levels[np.argmax(counts)]
    if best_attrib_type == 'disc':# make disc_splits dict
        disc_splits = dict()
        x_levels = np.unique(xdf[best_attrib])
        for l in x_levels:
            ind = xdf[best_attrib]==l
            disc_splits[l] = recursive_grow_tree(xdf[ind],ydf[ind],maxleaf,depth+1)
        root = tree_node(_type=best_attrib_type,disc_splits=disc_splits,attrib=best_attrib,leaf_cl=max_c,depth=depth)
    elif best_attrib_type == 'cont':# add left and right child
        root = tree_node(_type=best_attrib_type,cont_th=best_attrib_th,attrib=best_attrib,leaf_cl=max_c,depth=depth)

        ind = xdf[best_attrib] < best_attrib_th
        root.add_child(recursive_grow_tree(xdf[ind],ydf[ind],maxleaf,depth+1))#left child

        ind = xdf[best_attrib] >= best_attrib_th
        root.add_child(recursive_grow_tree(xdf[ind],ydf[ind],maxleaf,depth+1))#right child
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
        y_cond = Y[np.where(Xj==x)]
        y_levels, counts = np.unique(y_cond, return_counts=True)
        ly = len(y_cond)

        ps = [c/ly for c in counts]
        H += p_x[i] * entropy(ps)

    y_levels, counts = np.unique(Y, return_counts=True)
    ly = len(Y)
    ps = [c/ly for c in counts]
    return entropy(ps) - H

def mutual_information_j_continuous(Y, Xj):# takes less than 1 ms
    th = np.median(Xj)

    if type(Y) != np.ndarray:
        Y = np.array(Y)
    if type(Xj) != np.ndarray:
        Xj = np.array(Xj)

    lx = len(Xj)
    H = 0

    y_less = Y[np.where(Xj<th)]
    px = np.sum(Xj<th)/lx
    _, counts = np.unique(y_less, return_counts=True)
    ly = len(y_less)
    ps = [c/ly for c in counts]
    H += px * entropy(ps)

    y_great = Y[np.where(Xj>=th)]
    px = np.sum(Xj>=th)/lx
    _, counts = np.unique(y_great, return_counts=True)
    ly = len(y_great)
    ps = [c/ly for c in counts]
    H += px * entropy(ps)

    ly = len(Y)
    y_levels, counts = np.unique(Y, return_counts=True)
    ps = [c/ly for c in counts]
    return entropy(ps) - H, th

@silentdecorator(True)
def test(root, xdf, ydf):#generalize to go down all the testing paths, return all paths from disc_splits and then return the class of the argmax by weights. store number of samples in leaf
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

@silentdecorator(True)
def test2(root, xdf, ydf):#generalize to go down all the testing paths, return all paths from disc_splits and then return the class of the argmax by weights. store number of samples in leaf
    def treenodectr(l):
        c = 0
        if len(l) == 0:
            if type(l) == tree_node:
                c += 1
        else:
            for k in l:
                if type(k) == tree_node:
                    c += 1
        return c

    ypred = []
    ytrue = []
    for i in range(len(xdf)):
        x = xdf.iloc[i,:]
        y = root.decide(x,test2=True) # NOW OUR DECIDE FUNCTION MAY RETURN A LIST
        tn = treenodectr(y)
        pc = dict()
        deb = True
        while(tn > 0):
            if type(y) == list:
                if deb and type(y) == list:
                    deb = False
                    print("Unseen category found",file=sys.stderr) 
                y_new = []
                tn = 0
                for n in y:
                    if type(n) == tree_node:
                        yy = n.decide(x,test2=True)
                        y_new.append(yy)
                        if type(yy) == tree_node:
                            tn += 1
                        else:
                            if yy not in pc:
                                pc[yy] = n.num_samples
                            else:
                                pc[yy] += n.num_samples
                    else:
                        y_new.append(n)
                y = y_new
            else:
                y = y.decide(x,test2=True)
                if type(y) != tree_node and type(y) != list:
                    tn = 0
                elif type(y) == list:
                    tn = treenodectr(y)
        if type(y) != list:
            y_hat = y
        else:
            max_c = None
            max_s = -float('inf')
            for p in pc:
                if pc[p] > max_s:
                    max_s = pc[p]
                    max_c = p
            y_hat = max_c
        print(y_hat,ydf.iloc[i])
        ypred.append(y_hat)
        ytrue.append(ydf.iloc[i])
    return np.array(ypred), np.array(ytrue)

def count_nodes_upto_depth(root, d):
    nc = [0 for _ in range(d+1)]
    def cnud(a):
        nonlocal nc,d
        if a.depth <= d:
            for i in range(a.depth+1,d+1):
                nc[i] += 1
        elif a.depth > d:
            return
        if len(a.children) > 0:
            for r in a.children:
                cnud(r)
        elif a.disc_splits is not None and len(a.disc_splits) > 0:
            for r in a.disc_splits:
                cnud(a.disc_splits[r])
    cnud(root)
    return nc

@silentdecorator(True)
def get_acc_upto_depth(root, d, xdf, ydf):
    def gaud(d_l):
        nonlocal root,xdf,ydf
        ypred = []
        ytrue = []
        for i in range(len(xdf)):
            x = xdf.iloc[i,:]
            try:
                if d_l == 0:
                    ypred.append(root.leaf_cl)
                else:
                    y = root.decide(x)
                    while(type(y) == tree_node):
                        if y.depth < d_l:
                            y = y.decide(x)
                        else:
                            y = y.leaf_cl# this should break the loop
                    print(y,ydf.iloc[i])
                    ypred.append(y)
                ytrue.append(ydf.iloc[i])
            except AttributeNotFound:
                print("Sample has unidentified attributes, cannot process",file=sys.stderr)
        return np.array(ypred), np.array(ytrue)
    accs = OrderedDict()
    for d_l in range(d):
        y_hat,y = gaud(d_l)
        accs[d_l] = 100*np.sum(y_hat==y)/y.shape[0]
        print(accs[d_l],file=sys.stderr)
    return accs

def findmaxdepth(root):
    md = -float('inf')
    def rec(r):
        nonlocal md
        if r.depth > md:
            md = r.depth
        if len(r.children) > 0:
            for c in r.children:
                rec(c)
        elif r.disc_splits is not None and len(r.disc_splits) > 0:
            for c in r.disc_splits:
                rec(r.disc_splits[c])
    rec(root)
    return md

@timeitdecorator
def main():
    trainp = sys.argv[1]
    valp = sys.argv[2]
    testp = sys.argv[3]
    #############################################################
    df = pd.read_csv(trainp,delimiter=';')
    xdf = df.iloc[:,:-1]
    ydf = df.iloc[:,-1]

    root = growtree(xdf, ydf)
    max_depth = findmaxdepth(root)

    node_counts = count_nodes_upto_depth(root, max_depth+1)

    df = pd.read_csv(testp,delimiter=';')
    xdf_test = df.iloc[:,:-1]
    ydf_test = df.iloc[:,-1]

    df = pd.read_csv(valp,delimiter=';')
    xdf_val = df.iloc[:,:-1]
    ydf_val = df.iloc[:,-1]

    print("Train")
    train_accs = get_acc_upto_depth(root, max_depth+1, xdf, ydf)
    print("Test")
    test_accs = get_acc_upto_depth(root, max_depth+1, xdf_test, ydf_test)
    print("Val")
    val_accs = get_acc_upto_depth(root, max_depth+1, xdf_val, ydf_val)

    plt.plot(node_counts[1:],list(train_accs.values()),label='Training accuracies')
    plt.plot(node_counts[1:],list(test_accs.values()),label='Test accuracies')
    plt.plot(node_counts[1:],list(val_accs.values()),label='Validation accuracies')
    plt.xlabel('Number of nodes in the tree')
    plt.ylabel('Accuracy(in %)')
    plt.legend()
    plt.savefig('Accuracies_without_onehot.png')
    #plt.show()
    # y_hat,y = test2(root,xdf,ydf)
    # print(np.sum(y==y_hat)/y.shape[0])

if __name__ == "__main__":
    main()
