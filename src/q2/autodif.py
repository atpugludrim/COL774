from collections import defaultdict

class Variable:
    def __init__(this, value, local_gradients=()):
        this.val = value
        this.local_gradients = local_gradients
    def __add__(this, other):
        return add(this, other)
    def __mul__(this, other):
        return mul(this,other)
    def __sub__(this, other):
        return add(this,neg(other))
    def __truediv__(this, other):
        return mul(this, inv(other))

def add(a,b):
    value = a.val+b.val
    local_gradients = ((a,1),(b,1))
    return Variable(value,local_gradients)

def mul(a, b):
    val = a.val * b.val
    local_gradients = ((a,b.val),(b,a.val))
    return Variable(val, local_gradients)

def neg(a):
    val = -1*a.val
    local_gradients = ((a,-1))
    return Variable(val,local_gradients)

def inv(a):
    val = 1./a.val
    local_gradients = ((a,-1/a.value**2))
    return Variable(val,local_gradients)

def get_grads(var):
    grads = defaultdict(lambda: 0)
    def compute_grads(var, path_val):
        for child, lg in var.local_gradients:
            val_of_pth_to_child = path_val * lg
            grads[child] += val_of_pth_to_child
            compute_grads(child,val_of_pth_to_child)
    compute_grads(var, path_val=1)
    return grads

a = Variable(3)
b = Variable(2)
c = add(a,b)
d = mul(a,c)
grads = get_grads(d)
print(d.val,grads[a],grads[b],grads[c])
