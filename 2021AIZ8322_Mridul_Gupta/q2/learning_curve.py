import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def saveplt(J,col,name,lw):
    plt.figure()
    plt.plot([_ for _ in range(len(J))],J,color=col,lw=lw,linestyle="--")
    ax=plt.gca()
    ax.set_xlabel('Updates to theta')
    ax.set_ylabel('Cost')
    plt.savefig(name)

Js1 = np.load('Js_bs1.npy')
Js2 = np.load('Js_bs100.npy')
Js3 = np.load('Js_bs10000.npy')
Js4 = np.load('Js_bs1000000.npy')

lens = np.array([len(Js1),len(Js2),len(Js3),len(Js4)])
norm = plt.Normalize(min(lens),max(lens))
cols = cm.Wistia_r(norm(lens))
lw = [0.5,1,2,2]

for k in range(1,5):
    name = 'Js%d'%k
    saveplt(globals()[name],cols[0],name+".png",lw[k-1])
