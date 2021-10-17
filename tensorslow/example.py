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
# red_points = np.random.randn(50,2) - 2*np.ones((50,2))
# blue_points = np.random.randn(50,2) + 2*np.ones((50,2))
# plt.scatter(red_points[:,0],red_points[:,1],color='red')
# plt.scatter(blue_points[:,0],blue_points[:,1],color='blue')
# x_axis = np.linspace(-4,4,100)
# y_axis = -x_axis
# plt.plot(x_axis,y_axis)
# plt.show()
# X = placeholder()
# c = placeholder()
# W = Variable([
#     [1,-1],
#     [1,-1],
#     ])
# b = Variable([0,0])
# p = softmax(add(matmul(X,W),b))
# J = negative(reduce_sum(reduce_sum(multiply(c,log(p)),axis=1)))
# session = Session()
# print(session.run(J,{X:np.concatenate((blue_points,red_points)),c:[[1,0]]*len(blue_points)+[[0,1]]*len(red_points)}))
# output_probabilities = session.run(p,{
#     X: np.concatenate((blue_points,red_points))
#     })
# for op in output_probabilities:
#     print("{:.2f}, {:.2f}".format(op[0],op[1]))







# <LINEAR CLASSIFIER GRADIENT DESCENT>
# red_points = np.random.randn(50,2) - 2*np.ones((50,2))
# blue_points = np.random.randn(50,2) + 2*np.ones((50,2))
# X = placeholder()
# c = placeholder()
# 
# W = Variable(np.random.randn(2,2))
# b = Variable(np.random.randn(2))
# 
# p = softmax(add(matmul(X,W),b))
# 
# J = negative(reduce_sum(reduce_sum(multiply(c, log(p)), axis=1)))
# 
# minimization_op = GradientDescentOptimizer(learning_rate=0.01).minimize(J)
# 
# feed_dict = {
#         X: np.concatenate((blue_points, red_points)),
#         c:
#         [[1,0]]*len(blue_points)
#         + [[0,1]] * len(red_points)
#         }
# 
# session = Session()
# 
# for step in range(100):
#     J_value = session.run(J, feed_dict)
#     if step % 10 == 0:
#         print("Step:",step, " Loss:",J_value)
#     session.run(minimization_op, feed_dict)
# 
# W_value = session.run(W)
# print("Weigh matrix:\n",W_value)
# b_value = session.run(b)
# print("Bias:\n",b_value)
# 
# x_axis = np.linspace(-4, 4, 100)
# y_axis = -W_value[0][0]/W_value[1][0] * x_axis - b_value[0]/W_value[1][0]
# plt.plot(x_axis,y_axis)
# plt.scatter(red_points[:,0],red_points[:,1],color='red')
# plt.scatter(blue_points[:,0],blue_points[:,1],color='blue')
# plt.show()





# <MULTILAYER PERCEPTRON>
red_points = np.concatenate((
    0.2 * np.random.randn(25,2) + np.array([[0,0]]*25),
    0.2 * np.random.randn(25,2) + np.array([[1,1]]*25)
    ))
blue_points = np.concatenate((
    0.2 * np.random.randn(25,2) + np.array([[0,1]]*25),
    0.2 * np.random.randn(25,2) + np.array([[1,0]]*25)
    ))
# plt.scatter(red_points[:,0],red_points[:,1],color='red')
# plt.scatter(blue_points[:,0],blue_points[:,1],color='blue')
# plt.show()

X = placeholder()
c = placeholder() # CLASSES

W_hidden = Variable(np.random.randn(2,2))
b_hidden = Variable(np.random.randn(2))
p_hidden = sigmoid(add(matmul(X,W_hidden), b_hidden))

W_output = Variable(np.random.randn(2,2))
b_output = Variable(np.random.randn(2))
p_output = softmax(add(matmul(p_hidden, W_output), b_output))

J = negative(reduce_sum(reduce_sum(multiply(c, log(p_output)), axis=1)))
minimization_op = GradientDescentOptimizer(learning_rate=0.03).minimize(J)

feed_dict = {
        X: np.concatenate((blue_points, red_points)),
        c:
        [[1,0]]*len(blue_points)
        + [[0,1]] * len(red_points)
        }

session = Session()
steps = 1000
for step in range(steps):
    J_value = session.run(J, feed_dict)
    if step % 100 == 0:
        print("Step:",step," Loss:",J_value)
    session.run(minimization_op, feed_dict)

W_hidden_value = session.run(W_hidden)
print("Hidden layer weight matrix:\n", W_hidden_value)
b_hidden_value = session.run(b_hidden)
print("Hidden layer bias:\n",b_hidden_value)
W_output_value = session.run(W_output)
print("Hidden layer weight matrix:\n", W_output_value)
b_output_value = session.run(b_output)
print("Hidden layer bias:\n",b_output_value)

xs = np.linspace(-2,2)
ys = np.linspace(-2,2)
pred_classes = []

for x in xs:
    for y in ys:
        pred_class = session.run(p_output,
                feed_dict={X:[[x,y]]})[0]
        pred_classes.append((x,y,pred_class.argmax()))

xs_p, ys_p = [], []
xs_n, ys_n = [], []

for x, y, c in pred_classes:
    if c == 0:
        xs_n.append(x)
        ys_n.append(y)
    else:
        xs_p.append(x)
        ys_p.append(y)
fig, ax = plt.subplots(1,2)
ax[0].plot(xs_p, ys_p, 'r.', xs_n, ys_n, 'b.')
ax[1].scatter(red_points[:,0],red_points[:,1],color='red')
ax[1].scatter(blue_points[:,0],blue_points[:,1],color='blue')
ax[1].set_xlim(-2,2)
ax[1].set_ylim(-2,2)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_xticks([])
ax[1].set_yticks([])
plt.show()
