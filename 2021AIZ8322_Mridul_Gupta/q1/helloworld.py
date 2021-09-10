import numpy
import matplotlib.pyplot as plt
with open('/home/mridul/scai/ml/hw1/data/q1/linearX.csv','r') as f:
    _x = f.readlines()
with open('/home/mridul/scai/ml/hw1/data/q1/linearY.csv','r') as f:
    _y = f.readlines()

x = []
y = []
mean = 0
var = 0
n = 0
for i,j in zip(_x,_y):
    x.append(float(i.strip()))
    y.append(float(j.strip()))
    mean += x[-1]
    var += x[-1]*x[-1]
    n += 1

fig, ax = plt.subplots(1,2,figsize=(10,5))
ax[0].plot(x,y,'C0o',markersize=2)
var = (var - mean)/(n-1)
mean = mean/n

for i in range(len(x)):
    x[i] = (x[i]-mean)/var

ax[1].plot(x,y,'C1o',markersize=2)
plt.show()
