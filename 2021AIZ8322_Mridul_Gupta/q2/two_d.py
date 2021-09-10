import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size',required=True)
    parser.add_argument('--start-skip')
    parser.add_argument('--skip-frame')
    parser.add_argument('--sleep')
    parser.add_argument('-s',default=False,action='store_true')
    parser.add_argument('--stop-at')
    args = parser.parse_args()
    #############################################################
    batch_size = int(float(args.batch_size))
    filename = "thh_bs{}.npy".format(batch_size)
    with open(filename,'rb') as f:
        thh = np.load(f)
    x = thh[...,0]
    y = thh[...,1]
    z = thh[...,2]

    x_min, x_max = min(x),max(x)
    y_min, y_max = min(y),max(y)
    z_min, z_max = min(z),max(z)
    #############################################################
    if not args.s:
        plt.ion()
    ax = plt.axes(projection='3d')
    fig = plt.gcf()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel('th0')
    ax.set_ylabel('th1')
    ax.set_zlabel('th2')

    norm = plt.Normalize(z_min,z_max)
    cols = cm.turbo(norm(z))
    ax.view_init(azim=-138,elev=25)
    
    ###################  SETTING SOME OPTIONS  ###################
    flag = False
    if args.skip_frame is not None:
        flag = True
        sf = int(float(args.skip_frame))
        if args.start_skip is not None:
            ss = int(float(args.start_skip))
        else:
            ss = 300
    if args.sleep is not None:
        sd = float(args.sleep)
    else:
        sd = 1e-3
    #############################################################
    print("Total frames:",len(x))
    l = ax.plot([0],[0],[0],color=cols[0])
    pt = ax.plot([0],[0],[0],color=cols[0],linestyle="",marker='o')

    ctr = 0
    for i in range(len(x)):
        if flag and i > ss and i % sf:
            continue
        print(i,end="\r")
        l[0].set_data_3d(x[:i],y[:i],z[:i])
        pt[0].set_data_3d([x[i]],[y[i]],[z[i]])
        l[0].set_color(cols[i])
        pt[0].set_color(cols[i])
        fig.canvas.draw()
        fig.canvas.flush_events()
        if args.s:
            plt.savefig("frames/{:0>7d}_{:0>9d}.png".format(batch_size,ctr))
            ctr+=1
            if args.stop_at is not None and i > int(float(args.stop_at)):
                break
        else:
            time.sleep(sd)
    #############################################################
    #X,Z = np.random.rand(100)*100, np.random.rand(100)*100
    #norm = plt.Normalize(Z.min(),Z.max())
    #colors = cm.Pastel1_r(norm(Z))
    #for i, (x,z) in enumerate(zip(X,Z)):
    #    plt.plot(x,z,color=colors[i],marker='o')
    plt.show()

if __name__=="__main__":
    main()
