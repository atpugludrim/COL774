import sys
import numpy as np
import pandas as pd

# TODO: dataloader, sgd, operations, minimization op, session
train_path = '/home/anupam/Desktop/backups/COL774/data/q2/poker-hand-training-true.data'
test_path = '/home/anupam/Desktop/backups/COL774/data/q2/poker-hand-testing.data'

def main():
    df_train = pd.read_csv(train_path, header=None)
    # xdf_train = df_train.iloc[:,:-1]
    # ydf_train = df_train.iloc[:,-1]

    df_test = pd.read_csv(test_path, header=None)
    # xdf_test = df_test.iloc[:,:-1]
    # ydf_test = df_test.iloc[:,-1]

    df_train['train'] = 1
    df_test['train'] = 0
    df_comb = pd.concat([df_train,df_test])
    for c in df_comb.columns:
        if c != 'train':
            df_comb[c] = pd.Categorical(df_comb[c])
    dfc = pd.get_dummies(df_comb)
    df_train = dfc[dfc['train']==1]
    df_test = dfc[dfc['train']==0]
    df_train.drop(["train"], axis=1, inplace=True)
    df_test.drop(["train"], axis=1, inplace=True)

    # df_train = pd.concat([xdf_train,ydf_train],axis=1)
    # df_test = pd.concat([xdf_test,ydf_test],axis=1)
    # l_train = len(df_train)
    # l_val = int(0.05*l_train)
    # ind = np.random.permutation(len(df_test))
    # ind_val = ind[:l_val]
    # ind_test = ind[l_val:]
    # df_test2 = df_test.iloc[ind_test,:]
    # df_val = df_test.iloc[ind_val,:]

    df_train.to_csv('train.csv',header=False,index=False)
    df_test.to_csv('test.csv',header=False,index=False)
    # df_test2.to_csv('test.csv',header=False,index=False)
    # df_val.to_csv('val.csv',header=False,index=False)

if __name__=="__main__":
    main()
