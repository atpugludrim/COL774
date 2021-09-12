import argparse
import numpy as np
import pickle
from four_a import getdata, Scaler

def four_d(x, y):
    # STEP 1: FIND mu0, mu1, sig = sig0 = sig1, phi
    phi = np.sum(y) / len(y)

    m0 = np.sum(1-y)
    indicator_0 = np.repeat((1-y),2).reshape((y.shape[0],-1))
    mu0 = np.add.reduce(np.multiply(indicator_0,x/m0),axis=0)

    m1 = np.sum(y)
    indicator_1 = np.repeat(y,2).reshape((y.shape[0],-1))
    mu1 = np.add.reduce(np.multiply(indicator_1,x/m1),axis=0)

    z0 = np.multiply(indicator_0,(x-mu0)/np.sqrt(m0))
    sig0 = z0.T@z0
    z1 = np.multiply(indicator_1,(x-mu1)/np.sqrt(m1))
    sig1 = z1.T@z1

    return {'phi':phi, 'mu0':mu0, 'sig0':sig0, 'mu1':mu1, 'sig1':sig1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-x',required=True)
    parser.add_argument('--data-y',required=True)
    parser.add_argument('-scale',default=False,action='store_true')
    args = parser.parse_args()
    #############################################################
    x,y=getdata(args.data_x,args.data_y)
    if args.scale:
    #############################################################
        s0 = Scaler()
        s1 = Scaler()

        s0.fit(x[...,0].reshape((100,1)))
        s1.fit(x[...,1].reshape((100,1)))

        x[...,0] = s0.transform(x[...,0])
        x[...,1] = s1.transform(x[...,1])
    #############################################################
    th = four_d(x,y)
    with open('th_2.pkl','wb') as f:
        pickle.dump(th,f)

if __name__=="__main__":
    main()
