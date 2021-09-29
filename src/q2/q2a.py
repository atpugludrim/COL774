import numpy as np

def getdata(path):
    data_x = []
    data_y = []
    with open(path,"r") as f:
        for l in f:
            features = l.split(',')
            x = []
            for F in features[:-1]:
                x.append(int(F.strip()))
            data_x.append(x)
            data_y.append(y.strip())
    return np.array(data_x), np.array(data_y)

class Scaler:
    def __init__(this, mu, sig):
        this.mu = mu
        this.sig = sig
    def transform(this, data):
        return (data - this.mu)/this.sig
    def inv_transform(this, data):
        return (data * this.sig) + this.mu

def main():
    x,y = getdata()
    scaler = Scaler(x.mean,x.std)

if __name__=="__main__":
    main()
