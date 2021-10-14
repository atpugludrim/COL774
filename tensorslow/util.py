import numpy as np
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
