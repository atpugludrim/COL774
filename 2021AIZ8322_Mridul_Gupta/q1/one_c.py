import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import time
import argparse

from one_a import getdata, Scaler, J

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-x',required=True)
    parser.add_argument('--data-y',required=True)
    parser.add_argument('--no-rotate',default=False,action='store_true')
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

    th0 = np.linspace(-0.2,1,20)
    th1 = np.linspace(-0.5,0.5,20)
    th0, th1 = np.meshgrid(th0, th1)
    Z = np.empty(th0.shape)
    for i in range(th0.shape[0]):
        for j in range(th0.shape[1]):
            Z[i,j] = J(np.array([th0[i,j],th1[i,j]]), x, y)

    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.turbo(norm(1-Z))
    rcount, ccount, _ = colors.shape
    plt.ion() #TURNED OFF DURING SAVING FRAMES
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.plot_surface(th0,th1,Z,cmap='Greens',rstride=2,cstride=2,alpha=.8)
    #ax.plot_wireframe(th0,th1,Z,cmap='Greens',rstride=2,cstride=2,alpha=.8)
    surf = ax.plot_surface(th0,th1,Z,rcount=rcount,ccount=ccount,facecolors=colors,shade=False)
    surf.set_facecolor((0,0,0,0))
    f = ax.plot(thh[0][0],thh[0][1],Js[0],'ro')
    elev = 30.
    azim = -74.
    diff = 0
    ddiff = 0.2
    flag = args.no_rotate
    for t in range(1,len(Js)):
        if not flag:
            ax.view_init(elev=elev,azim = azim)
            azim = (-64+diff)%360
            if (diff+ddiff) > 20 or (diff+ddiff) < 0:
                ddiff = -ddiff
            diff = (diff+ddiff)
        #
        f[0].set_data_3d(thh[t][0],thh[t][1],Js[t])
        fig.canvas.draw()
        fig.canvas.flush_events()
        #plt.savefig("framessurf/{:0>5d}.png".format(t))
        time.sleep(0.001)
    plt.show()

if __name__=="__main__":
    main()
