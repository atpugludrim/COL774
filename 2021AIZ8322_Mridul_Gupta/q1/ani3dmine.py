import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

fps = 10
frn = 50

th0 = np.linspace(-30,30,50)
th1 = np.linspace(0,2,50)
th0, th1 = np.meshgrid(th0, th1)
with open('Costs.npy','rb') as f:
    z = np.load(f)
with open('thetas.npy','rb') as f:
    ths = np.load(f)
with open('Js.npy','rb') as f:
    Js = np.load(f)

def update_plot(frn, *args):
    print(args,frn)
    plot[1].remove()
    plot[1] = ax.scatter(ths[frn,0],ths[frn,1],Js[frn],color='r')

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.view_init(elev=-10.,azim=20)

plot = [ax.plot_surface(th0,th1,z,color='0.75',rstride=1,cstride=1,cmap='summer'),ax.scatter(ths[0,0],ths[0,1],Js[0],color='r')]
#ax.set_zlim(0,1.1)
ax.set_xlabel("th1")
ax.set_ylabel("th0")
ax.set_zlabel("J")
ani = animation.FuncAnimation(fig,update_plot,frn,fargs=(plot),interval=1000/fps)
fn = 'plotanim'
ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
