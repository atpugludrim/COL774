import matplotlib.pyplot as plt
import numpy as np
import time
import argparse

from one_a import getdata, Scaler, J

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-x',required=True)
    parser.add_argument('--data-y',required=True)
    args = parser.parse_args()
    #############################################################
    x, y = getdata(args.data_x,args.data_y)
    scaler = Scaler()
    scaler.fit(x)
    x = scaler.transform(x)

    with open("th_history_ref.npy","rb") as f:
        thh = np.load(f)
    with open("Js_ref.npy","rb") as f:
        Js = np.load(f)

    th0 = np.linspace(-0.2,1.1,20)
    th1 = np.linspace(-0.5,0.5,20)
    th0, th1 = np.meshgrid(th0, th1)
    Z = np.empty(th0.shape)
    for i in range(th0.shape[0]):
        for j in range(th0.shape[1]):
            Z[i,j] = J(np.array([th0[i,j],th1[i,j]]), x, y)

    plt.ion()
    fig = plt.figure()
    ax = fig.gca()
    ax.contour(th0,th1,Z,20,cmap='turbo')
    f = ax.plot(thh[0][0],thh[0][1],'rx', markersize=20)
    l = len(Js)
    s = 1e-5
    modv = 10
    for t in range(1,l):
        if modv == 10 and t > 1400:
            modv = 20
        if t > 200 and t % modv:
            continue
        print("Frame: [",t,"/",l,"]",end="\r")
        f[0].set_data(thh[t][0],thh[t][1])
        fig.canvas.draw()
        fig.canvas.flush_events()
        #plt.savefig("framescontour/{:0>5d}.png".format(t))
        time.sleep(s)
    plt.show()
    print("\n")

if __name__=="__main__":
    main()
