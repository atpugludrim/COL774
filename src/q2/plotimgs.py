import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def openfile(path):
    df = pd.read_csv(path,header=None).astype(dtype=np.int)
    return df.values.reshape(-1)

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-true",type=str,required=True)
    parser.add_argument("-pred",type=str,required=True)
    args = parser.parse_args()
    return args

def get_misclassified_indices(t,p):
    return np.where((t!=p)==True)[0]

def extractimg(l):
    x = [int(t.strip()) for t in l.strip().split(',')]
    x = x[:-1]
    return x

def getimages(chosen):
    ctr = 0
    imgs = []
    with open('/home/mridul/scai/ml/hw2/data/q2/test.csv','r') as f:
        for i, l in enumerate(f): 
            if i in chosen:
                ctr += 1
                img = extractimg(l)
                imgs.append(img)
            if ctr == len(chosen):
                break
    return np.array(imgs)

def main():
    args = getargs()
    true = openfile(args.true)
    pred = openfile(args.pred)
    ind = get_misclassified_indices(true,pred)
    perm = np.random.permutation(ind.shape[0])
    chosen = np.sort(ind[perm[:10]])
    imgs = getimages(chosen)

    fig, ax = plt.subplots(2,5)
    ctr = 0
    for i in range(2):
        for j in range(5):
            ax[i][j].imshow(imgs[ctr].reshape(28,28),cmap='gray')
            p = pred[chosen[ctr]]
            t = true[chosen[ctr]]
            ax[i][j].set_title(f"P{p}:T{t}")
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            #
            ctr += 1
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
