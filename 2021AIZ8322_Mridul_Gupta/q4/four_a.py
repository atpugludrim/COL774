import numpy as np
import pickle
import argparse

class Scaler:
    def __init__(this, mu=0, sig_sq=1):
        this.mu = mu
        this.sig_sq = sig_sq

    def transform(this, data):
        return (data - this.mu) / np.sqrt(this.sig_sq)

    def inv_tr(this, data):
        return (data * np.sqrt(this.sig_sq)) + this.mu

    def fit(this, data):
        mean = np.zeros(data.shape[1])
        var = np.zeros(data.shape[1])
        n = data.shape[0]

        for x in data:
            mean += x/n
            var += np.multiply(x,x)/(n-1)

        var = var - mean*(n-1)/n
        this.mu = mean
        this.sig_sq = var

def getdata(dx,dy):
    with open(dx,'r') as f:
        _x = f.readlines()
    with open(dy,'r') as f:
        _y = f.readlines()

    data = np.array([])
    labels = np.array([])
    classes = dict()
    code = 0
    for x, y in zip(_x, _y):
        data = np.append(data, [float(k.strip()) for k in x.split()])
        lab = y.strip()
        if lab not in classes:
            classes[lab] = code
            code += 1
        labels = np.append(labels, classes[lab])

    return (data.reshape((len(data)//2,2)), labels)

def four_a(x, y):
    # STEP 1: FIND mu0, mu1, sig = sig0 = sig1, phi
    phi = np.sum(y) / len(y)

    m0 = np.sum(1-y)
    indicator_0 = np.repeat((1-y),2).reshape((y.shape[0],-1))
    mu0 = np.add.reduce(np.multiply(indicator_0,x/m0),axis=0)

    m1 = np.sum(y)
    indicator_1 = np.repeat(y,2).reshape((y.shape[0],-1))
    mu1 = np.add.reduce(np.multiply(indicator_1,x/m1),axis=0)

    z0 = np.multiply(indicator_0,(x-mu0)/np.sqrt(len(y)))
    sig0 = z0.T@z0
    z1 = np.multiply(indicator_1,(x-mu1)/np.sqrt(len(y)))
    sig1 = z1.T@z1

    sig = sig0+sig1

    return {'phi':phi, 'mu0':mu0, 'sig':sig, 'mu1':mu1}

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
    th = four_a(x,y)
    with open('th.pkl','wb') as f:
        pickle.dump(th,f)

if __name__=="__main__":
    main()
