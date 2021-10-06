import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=str)
    args = parser.parse_args()
    with open("true.pkl","rb") as f:
        true = pickle.load(f)
    with open("pred_1a.pkl","rb") as f:
        pred = pickle.load(f)
    cm = [[0 for _ in range(5)] for _ in range(5)]
    for t,p in zip(true,pred):
        cm[t-1][p-1] += 1
    rowLabs = ["Actual {}  ".format(i+1) for i in range(5)]
    colLabs = ["Predicted {}".format(i+1) for i in range(5)]
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
    plt.gca().set_axis_off()
    plt.savefig('{}.png'.format(args.output))
    plt.show()
if __name__=="__main__":
    main()
