import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

class DataLoader:
    def __init__(this, x, y, bs):
        this.x = x
        this.y = y
        this.bs = bs
        this.ln = len(x)
    def get_iterator(this):
        end_ind = 0
        ln = this.ln
        def getit():
            nonlocal end_ind,ln
            while end_ind < ln:
                start_ind = end_ind
                end_ind = min(start_ind+this.bs,ln)
                yield this.x[start_ind:end_ind],this.y[start_ind:end_ind]
        return getit()
class timer:
    def __init__(this,string):
        this.string=string
    def __enter__(this):
        this.s = time.time()
    def __exit__(this,a,b,c):
        print("It took {:.3f}s to".format(time.time() - this.s),this.string)
def make_conf_mat(y_hat, y, filename):
    fig = plt.figure()
    cm = [[0 for _ in range(10)] for _ in range(10)]
    for t,p in zip(y,y_hat):
        cm[int(t)][int(p)] += 1
    rowLabs = ["Actual {}  ".format(i) for i in range(10)]
    colLabs = ["P {}".format(i) for i in range(10)]
    rcolors = plt.cm.BuPu(np.full(len(rowLabs),0.1))
    ccolors = plt.cm.BuPu(np.full(len(colLabs),0.1))
    celtext = []
    for r in cm:
        celtext.append([str(c) for c in r])
    table = plt.table(cellText=celtext,
            rowLabels=rowLabs,
            colLabels=colLabs,
            cellLoc='center',
            loc='upper left',
            rowColours=rcolors,
            colColours=ccolors,
            )
    table.scale(1,2)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.gca().set_axis_off()
    # plt.show()
    plt.savefig('{}'.format(filename))
    plt.close(fig)
def test(clf):
    global dataloader
    acc = 0
    ln = dataloader.ln
    y_tru = []
    y_pre = []
    for x, y in dataloader.get_iterator():
        y_hat = clf.predict(x)
        yy_hat = np.argmax(y_hat,axis=1)
        yy = np.argmax(y,axis=1)
        acc += np.sum(yy_hat==yy)/ln
        y_tru.extend(list(yy))
        y_pre.extend(list(yy_hat))
    return acc*100, y_pre, y_tru
def main():
    with timer("load train data"):
        df = pd.read_csv('train.csv')
        x_train = df.iloc[:,:-10].values
        y_train = df.iloc[:,-10:].values
    
    with timer("load test data"):
        df = pd.read_csv('test.csv')
        x_test = df.iloc[:,:-10].values
        y_test = df.iloc[:,-10:].values

    with timer("make dataloader"):
        global dataloader
        dataloader = DataLoader(x_test,y_test,5000)

    clf = MLPClassifier(hidden_layer_sizes=(100,100),solver='sgd',batch_size=100,learning_rate_init=0.1,validation_fraction=0,alpha=0,learning_rate='invscaling',power_t=0.5)
    with timer("train sklearn's mlp"):
        clf.fit(x_train, y_train)
    with timer("test mlp"):
        acc,y_pre,y_tru=test(clf)
    print(acc)
    make_conf_mat(y_pre,y_tru,"conf_mat_sklearn_relu")

if __name__=="__main__":
    main()
