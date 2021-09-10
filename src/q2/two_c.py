import numpy as np
import argparse

def two_c(ths,tf,nsf):
    m = 0
    x = []
    y = []
    y_hat = []
    ################## <HYPOTHESIS> ###################
    hthx = lambda x: ths[0] + ths[1] * x[0] + ths[2] * x[1]
    ################# </HYPOTHESIS> ###################
    err = 0
    with open(tf,'r') as testfile:
        for i,l in enumerate(testfile):
            if i == 0 and not nsf:
                continue
            m += 1
            token = l.strip().split(',')
            ###################################################
            x.append([float(t.strip()) for t in token[:-1]])
            y.append(float(token[-1].strip()))
            y_hat.append(hthx(x[-1]))
            ###################################################
            err = err * ((m - 1)/m) + np.square(y[-1] - y_hat[-1])/m
            #################  ^^TO AVOID OVERFLOW  ###########
    return err/2
def main():
    ###################################################
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch-size",required=True)
    parser.add_argument("--test-file",required=True)
    parser.add_argument("--no-skip-first",default=False,action="store_true")
    args = parser.parse_args()
    ###################################################
    batch_size = int(float(args.batch_size))
    with open("ths_bs{}.npy".format(batch_size),'rb') as f:
        ths = np.load(f)
    ths = np.array([3,1,2])
    err = two_c(ths,args.test_file,args.no_skip_first)
    for i,t in enumerate(ths):
        print("th{}:\t{}".format(i,t))
    print("MSE is {}.".format(err))

if __name__=="__main__":
    main()
