import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=str)
    args = parser.parse_args()
    #############################################################
    true_ = pd.read_csv('true_multiclass.csv').astype(dtype=np.int)
    true = true_.values.reshape(-1)
    pred_ = pd.read_csv('pred_multiclass.csv').astype(dtype=np.int)
    pred = pred_.values.reshape(-1)
    #############################################################
    # with open("true_multi","rb") as f:
    #     true = pickle.load(f)
    # with open("pred.pkl","rb") as f:
    #     pred = pickle.load(f)
    cm = [[0 for _ in range(10)] for _ in range(10)]
    for t,p in zip(true,pred):
        cm[t][p] += 1
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
    table.set_fontsize(8)
    plt.gca().set_axis_off()
    plt.savefig('{}.png'.format(args.output))
    plt.show()
if __name__=="__main__":
    main()
