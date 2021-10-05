import sys
import pickle
import numpy as np

def main():
    with open("true.pkl","rb") as f:
        true = pickle.load(f)
    with open(sys.argv[1],"rb") as f:
        pred = pickle.load(f)
    cm = [[0 for _ in range(5)] for _ in range(5)]
    for t,p in zip(true,pred):
        cm[t-1][p-1] += 1
    cm = np.array(cm)

    F1 = np.zeros(5)

    for class_ in range(5):
        pr = cm[class_,class_]/np.sum(cm[:,class_])
        rc = cm[class_,class_]/np.sum(cm[class_,:])
        if pr == 0 and rc == 0:
            F1[class_] = 0
        else:
            F1[class_] = 2*pr*rc/(pr+rc)
        print(f"F1 score for {class_+1} is {F1[class_]}")
    macro_F1 = F1.mean()
    print(f"Macro F1 score is {macro_F1}")

if __name__=="__main__":
    main()
