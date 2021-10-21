import numpy as np
from queue import Queue

class Op:
    def __init__(this, inp = []):
        this.inp = inp
        this.cons = []
        for inp_n in this.inp:
            inp_n.cons.append(this)

        _default_graph.ops.append(this)

    def compute(this):
        pass

class add(Op):
    def __init__(this, x, y):
        super().__init__([x,y])
    def compute(this, x_val, y_val):
        return x_val + y_val

class matmul(Op):
    def __init__(this, a, b):
        super().__init__([a,b])
    def compute(this, a_val, b_val):
        return a_val.dot(b_val)

class placeholder:
    def __init__(this):
        this.cons = []
        _default_graph.placeholders.append(this)

class variable:
    def __init__(this, init_val = None):
        this.val = init_val
        this.cons = []
        _default_graph.variables.append(this)

class Graph:
    def __init__(this):
        this.ops = []
        this.variables = []
        this.placeholders = []
    def asdefault(this):
        global _default_graph
        _default_graph = this

class Session:
    def traverse_postorder(this, op):
        nodes_postorder = []
        def rec(node):
            if isinstance(node, Op):
                for inp_n in node.inp:
                    rec(inp_n)
            nodes_postorder.append(node)
        rec(op)
        return nodes_postorder

    def run(this, op, feed_dict={}):
        nodes_postorder = this.traverse_postorder(op)
        for n in nodes_postorder:
            if type(n) == placeholder:
                n.output = feed_dict[n]
            elif type(n) == variable:
                n.output = n.val
            else:
                n.inputs = [i.output for i in n.inp] # concrete values, not abstract nodes
                n.output = n.compute(*n.inputs)
            if type(n.output) == list:
                n.output = np.array(n.output)
        return op.output

class sigmoid(Op):
    def __init__(this, z):
        super().__init__([z])
    def compute(this, z_val):
        return 1/(1+np.exp(-z_val))

class softmax(Op):
    def __init__(this, z):
        super().__init__([z])
    def compute(this, z_val):
        return np.exp(z_val)/np.sum(np.exp(z_val),axis=1)[:,None]

class log(Op):
    def __init__(this, c):
        super().__init__([c])
    def compute(this, c_Val):
        return np.log(c_Val)

class multiply(Op):
    def __init__(this, a, b):
        super().__init__([a,b])
    def compute(this, a_val, b_val):
        return a_val * b_val

class reduce_sum(Op):
    def __init__(this, A, axis=None):
        super().__init__([A])
        this.axis = axis
    def compute(this, A_val):
        return np.sum(A_val, axis=this.axis)

class negative(Op):
    def __init__(this, b):
        super().__init__([b])
    def compute(this, b_val):
        return -b_val

class SGDOptimizer:
    def __init__(this, lr):
        this.lr = lr
    def minimize(this, loss):
        lr = this.lr

        class MinimizationOperation(Op):
            def compute(this):
                grad_table = compute_gradients(loss)

                for n in grad_table:
                    if type(n) == variable:
                        grad = grad_table[n]
                        n.val -= lr * grad
        return MinimizationOperation()

_grad_reg = {}

def RegisterGradient(fname):
    def decorator(func):
        _grad_reg[eval(fname)] = func
        return func
    return decorator

def compute_gradients(loss):
    gt = {}
    gt[loss] = 1
    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        n = queue.get()
        if n != loss:
            gt[n] = 0 # accumulator
            for c in n.cons:
                lgwco = gt[c]
                cc = c.__class__
                bprop = _grad_reg[cc]
                lgwci = bprop(c, lgwco)

                if len(c.inp) == 1:
                    gt[n] += lgwci
                else:
                    niici = c.inp.index(n)
                    lgwn = lgwci[niici]
                    gt[n] += lgwn
        if hasattr(n, "inp"):
            for inp_n in n.inp:
                if not inp_n in visited:
                    visited.add(inp_n)
                    queue.put(inp_n) # bfs
    return gt

@RegisterGradient("negative")
def neg_pr(op, grad):
    return -grad

@RegisterGradient("log")
def log_pr(op, grad):
    return grad/op.inputs[0]

@RegisterGradient("sigmoid")
def sig_pr(op, grad):
    return grad * op.output * (1 - op.output)

@RegisterGradient("multiply")
def mul_pr(op, grad):
    a = op.inputs[0]
    b = op.inputs[1]
    return [grad*b,grad*a]

@RegisterGradient("matmul")
def matmul_pr(op, gr):
    a = op.inputs[0]
    b = op.inputs[1]
    return [gr.dot(b.T), a.T.dot(gr)]

@RegisterGradient("add")
def add_pr(op, gr):
    a = op.inputs[0]
    b = op.inputs[1]

    gra = gr
    grb = gr

    while np.ndim(gra) > len(a.shape):
        gra = np.sum(gra,axis=0)
    for ax, s in enumerate(a.shape):
        if s == 1:
            gra = np.sum(gra, axis=ax, keep_dims=True)

    while np.ndim(grb) > len(b.shape):
        grb = np.sum(grb,axis=0)
    for ax, s in enumerate(b.shape):
        if s == 1:
            grb = np.sum(grb, axis=ax, keep_dims=True)

    return [gra, grb]

@RegisterGradient("reduce_sum")
def redsm_pr(op, gr):
    a = op.inputs[0]

    out_sh = np.array(a.shape)
    out_sh[op.axis] = 1

    t = a.shape//out_sh
    gr = np.reshape(gr, out_sh)
    return np.tile(gr, t)

@RegisterGradient("softmax")
def sm_pr(op,gr):
    sm = op.output
    return (gr - np.reshape(np.sum(gr*sm,1),[-1,1]))*sm
