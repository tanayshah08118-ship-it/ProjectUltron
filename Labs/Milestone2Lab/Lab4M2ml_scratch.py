import random

class Value:
    def __init__(self,data,_children=(),_op=""):
        self.data=data
        self.grad=0.0
        self._prev=set(_children)
        self._op=_op
        self._backward= lambda: None
    def __add__(self,other):
        other=other if isinstance(other,Value)else Value(other)
        out = Value(self.data+other.data,(self,other),_op="+")
        def _backward():
            self.grad+=1.0*out.grad
            other.grad+=1.0*out.grad
        out._backward=_backward
        return out
    def backward(self):
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad=1.0
        for node in reversed(topo):
            node._backward()
    def __mul__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out=Value(self.data*other.data,(self,other),"*")
        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backward
        return out
    def ReLU(self):
        out = Value(0.0 if self.data<0 else self.data,(self,),"ReLU")
        def _backward():
            self.grad+=(out.data>0)*out.grad
        out._backward=_backward
        return out
class Neurons:
    def __init__(self,nin,nonlin=True):
        self.w= [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b= Value(random.uniform(-1,1))
        self.nonlin=nonlin
    def __call__(self,x):
        act = sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        return act.ReLU() if self.nonlin else act
    def parameter(self):
        return self.w+[self.b]
class Layers:
    def __init__(self,nin,nout,**kwargs):
        self.neurons=[Neurons(nin,**kwargs) for _ in range(nout)]
    def __call__(self,x):
        outs =[n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs
    def parameter(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameter())
        return params
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layers(sz[i], sz[i+1], nonlin=(i != len(nouts) - 1)) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameter(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameter())
        return params
print("-" * 60)
print("EXPERIMENT A: Inducing Total Brain Death")
print("-" * 60)

# 1. Create a model where the output layer ALSO has ReLU
# (Notice: pass nonlin=True to all layers!)
model = MLP(2, [4, 4, 1])
# model.layers[-1].neurons[0].nonlin = True  # Force output neuron to use ReLU! <--TOTAL BRAIN DEATH CONFIRMED!  
model.layers[-1].neurons[0].nonlin = False
# 2. Set the output neuron's bias to a strong negative number
model.layers[-1].neurons[0].b = Value(-5.0)

# 3. Forward pass
x = [Value(2.0), Value(3.0)]
out = model(x)
print("Model Output:", out.data)

# 4. Backward pass
out.backward()

# 5. Count how many parameters received a non-zero gradient
active_grads = [p for p in model.parameter() if p.grad != 0.0]
print(f"Active Parameters: {len(active_grads)} / {len(model.parameter())}")