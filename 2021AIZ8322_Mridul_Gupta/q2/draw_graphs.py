import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from two_b import getdata

def main():
    plotSample = False
    with open("x.npy","rb") as f:
        x = np.load(f)
    with open("y.npy", "rb") as f:
        y = np.load(f)

    x1 = np.linspace(-1,6,20)
    x2 = np.linspace(-4,3,20)
    x1, x2 = np.meshgrid(x1, x2)
    y_ = np.empty(x1.shape)

    with open("ths.npy","rb") as f:
        ths = np.load(f)

    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            y_[i,j] = ths[0]+ths[1]*x1[i,j]+ths[2]*x2[i,j]

    n_points = 50
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x1,x2,y_,rstride=2,cstride=2,cmap="Greens")
    ax.scatter(x[:n_points,0],x[:n_points,1],y[:n_points],color="C3",s=5)
    if plotSample:
        x_sample,y_sample = getdata(5*1e1)
        ax.scatter(x_sample[:,0],x_sample[:,1],y_sample,color="C0",s=5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.show()
    #############################################################

    with open("Js.npy",'rb') as f:
        Js = np.load(f)

    N = Js.shape[0]
    average_over = []
    # for k in range(Js.shape[0]):
    for k in range(N//500):
        try:
            average_over.append(np.mean(Js[k:k+500]))
        except Exception as e:
            print("Exception",e)
            pass
    plt.plot([_ for _ in range(len(average_over))],average_over,'k--')
    plt.show()

if __name__ == "__main__":
    main()
