import numpy as np
from queue import Queue
class Operation:
    def __init__(this, input_nodes = []):
        this.input_nodes = input_nodes
        this.consumers = []
        for input_node in input_nodes:
            input_node.consumers.append(this)

        _default_graph.operations.append(this)

    def compute(this):
        pass

class add(Operation):
    def __init__(this, x, y):
        super().__init__([x, y])
    def compute(this, x_value, y_value):
        return x_value + y_value

class matmul(Operation):
    def __init__(this, a, b):
        super().__init__([a,b])
    def compute(this, a_value, b_value):
        return a_value.dot(b_value)

class placeholder:
    def __init__(this):
        this.consumers = []
        _default_graph.placeholders.append(this)

class Variable:
    def __init__(this, initial_value=None):
        this.value = initial_value
        this.consumers = []
        _default_graph.variables.append(this)

class Graph:
    def __init__(this):
        this.operations = []
        this.placeholders = []
        this.variables = []

    def as_default(this):
        global _default_graph
        _default_graph = this

class Session:
    def run(this, operation, feed_dict={}):
        nodes_postorder = traverse_postorder(operation)
        for node in nodes_postorder:
            if type(node) == placeholder:
                node.output = feed_dict[node]
            elif type(node) == Variable:
                node.output = node.value
            else:
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)

        return operation.output

def traverse_postorder(operation):
    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder

class sigmoid(Operation):
    def __init__(this, z):
        super().__init__([z])
    def compute(this, z_val):
        return 1/(1+np.exp(-z_val))

class softmax(Operation):
    def __init__(this, z):
        super().__init__([z])
    def compute(this, z_val):
        return np.exp(z_val)/np.sum(np.exp(z_val),axis=1)[:,None]

class log(Operation):
    def __init__(this, x):
        super().__init__([x])
    def compute(this, x_val):
        return np.log(x_val)

class multiply(Operation):
    def __init__(this, x, y):
        super().__init__([x,y])
    def compute(this, x_val, y_val):
        return x_val*y_val

class reduce_sum(Operation):
    def __init__(this, A, axis=None):
        super().__init__([A])
        this.axis = axis
    def compute(this, A_val):
        return np.sum(A_val, this.axis)

class negative(Operation):
    def __init__(this, x):
        super().__init__([x])
    def compute(this, x_val):
        return -x_val

class GradientDescentOptimizer:
    def __init__(this, learning_rate):
        this.learning_rate = learning_rate

    def minimize(this, loss):
        lr = this.learning_rate

        class MinimizationOperation(Operation):
            def compute(this):
                grad_table = compute_gradients(loss)
                
                for node in grad_table:
                    if type(node) == Variable:
                        grad = grad_table[node]
                        node.value -= lr * grad

        return MinimizationOperation()

_gradient_registry = {}

class RegisterGradient:
    def __init__(this, op_type):
        this._op_type = eval(op_type)

    def __call__(this, f):
        _gradient_registry[this._op_type] = f
        return f

def compute_gradients(loss):
    grad_table = {}
    grad_table[loss] = 1
    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        node = queue.get()
        if node != loss:
            grad_table[node] = 0
            for consumer in node.consumers:
                lossgrad_wrt_consumer_output = grad_table[consumer]
                consumer_op_type = consumer.__class__
                bprop = _gradient_registry[consumer_op_type]

                lossgrads_wrt_consumer_inputs = bprop(consumer, lossgrad_wrt_consumer_output)
                if len(consumer.input_nodes) == 1:
                    grad_table[node] += lossgrads_wrt_consumer_inputs
                else:
                    node_index_in_consumer_inputs = consumer.input_nodes.index(node)
                    lossgrad_wrt_node = lossgrads_wrt_consumer_inputs[node_index_in_consumer_inputs]
                    grad_table[node] += lossgrad_wrt_node
        if hasattr(node, "input_nodes"):
            for input_node in node.input_nodes:
                if not input_node in visited:
                    visited.add(input_node)
                    queue.put(input_node)
    return grad_table

@RegisterGradient("negative")
def _negative_gradient(op, grad):
    return -grad

@RegisterGradient("log")
def _log_gradient(op, grad):
    x = op.inputs[0]
    return grad/x

@RegisterGradient("sigmoid")
def _sigmoid_gradient(op, grad):
    sigmoid = op.output
    return grad * sigmoid * (1 - sigmoid)

@RegisterGradient("multiply")
def _multiply_gradient(op, grad):
    A = op.inputs[0]
    B = op.inputs[1]
    return [grad*B, grad*A]

@RegisterGradient("matmul")
def _matmul_gradient(op, grad):
    A = op.inputs[0]
    B = op.inputs[1]
    return [grad.dot(B.T), A.T.dot(grad)]

@RegisterGradient("add")
def _add_gradient(op, grad):
    a = op.inputs[0]
    b = op.inputs[1]

    grad_wrt_a = grad
    grad_wrt_b = grad

    while np.ndim(grad_wrt_a) > len(a.shape):
        grad_wrt_a = np.sum(grad_wrt_a,axis=0)
    for axis, size in enumerate(a.shape):
        if size == 1:
            grad_wrt_a = np.sum(grad_wrt_a, axis=axis, keepdims=True)

    while np.ndim(grad_wrt_b) > len(b.shape):
        grad_wrt_b = np.sum(grad_wrt_b, axis = 0)

    for axis, size in enumerate(b.shape):
        if size == 1:
            grad_wrt_b = np.sum(grad_wrt_b, axis=axis, keepdims=True)

    return [grad_wrt_a, grad_wrt_b]

@RegisterGradient("reduce_sum")
def _reduce_sum_gradient(op, grad):
    A = op.inputs[0]

    output_shape = np.array(A.shape)
    output_shape[op.axis] = 1
    tile_scaling = A.shape // output_shape
    grad = np.reshape(grad, output_shape)
    return np.tile(grad, tile_scaling)

@RegisterGradient("softmax")
def _softmax_gradient(op, grad):
    softmax = op.output
    return (grad - np.reshape(
        np.sum(grad * softmax, 1),
        [-1, 1]
        )) * softmax
