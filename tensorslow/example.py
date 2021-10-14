from util import *
import matplotlib.pyplot as plt
import numpy as np

Graph().as_default()

# A = Variable([[1,0],[0,-1]])
# b = Variable([[1,1]])
# x = placeholder()
# 
# y = matmul(A,x)
# 
# z = add(y,b)
# </EX1>
# x = placeholder()
# w = Variable([1,1])
# b = Variable(0)
# p = sigmoid(add(matmul(w,x),b))
# session = Session()
# output = session.run(p,{x:[3,2]})
# print(output)
# </EX2>
red_points = np.random.randn(50,2) - 2*np.ones((50,2))
blue_points = np.random.randn(50,2) + 2*np.ones((50,2))
plt.scatter(red_points[:,0],red_points[:,1],color='red')
plt.scatter(blue_points[:,0],blue_points[:,1],color='blue')
x_axis = np.linspace(-4,4,100)
y_axis = -x_axis
plt.plot(x_axis,y_axis)
plt.show()
X = placeholder()
c = placeholder()
W = Variable([
    [1,-1],
    [1,-1],
    ])
b = Variable([0,0])
p = softmax(add(matmul(X,W),b))
J = negative(reduce_sum(reduce_sum(multiply(c,log(p)),axis=1)))
session = Session()
print(session.run(J,{X:np.concatenate((blue_points,red_points)),c:[[1,0]]*len(blue_points)+[[0,1]]*len(red_points)}))
# output_probabilities = session.run(p,{
#     X: np.concatenate((blue_points,red_points))
#     })
# for op in output_probabilities:
#     print("{:.2f}, {:.2f}".format(op[0],op[1]))
