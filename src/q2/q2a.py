import sys
import numpy as np
import pandas as pd

# TODO: dataloader, sgd, operations, minimization op, session
train_path = '/home/anupam/Desktop/backups/COL774/data/q2/poker-hand-training-true.data'
test_path = '/home/anupam/Desktop/backups/COL774/data/q2/poker-hand-testing.data'

def main():
    df_train = pd.read_csv(train_path, header=None)
    xdf_train = df_train.iloc[:,:-1]
    ydf_train = df_train.iloc[:,-1]

    df_test = pd.read_csv(test_path, header=None)
    xdf_test = df_test.iloc[:,:-1]
    ydf_test = df_test.iloc[:,-1]

    xdf_train['train'] = 1
    xdf_test['train'] = 0
    xdf_comb = pd.concat([xdf_train,xdf_test])
    for c in xdf_comb.columns:
        if c != 'train':
            xdf_comb[c] = pd.Categorical(xdf_comb[c])
    dfc = pd.get_dummies(xdf_comb)
    xdf_train = dfc[dfc['train']==1]
    xdf_test = dfc[dfc['train']==0]
    xdf_train.drop(["train"], axis=1, inplace=True)
    xdf_test.drop(["train"], axis=1, inplace=True)

    df_train = pd.concat([xdf_train,ydf_train],axis=1)
    df_test = pd.concat([xdf_test,ydf_test],axis=1)

    df_train.to_csv('train.csv',header=False,index=False)
    df_test.to_csv('test.csv',header=False,index=False)

if __name__=="__main__":
    main()
