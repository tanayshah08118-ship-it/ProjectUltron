import random

"""Building Your Own Autograd Engine"""

#Initilazing the CLASS and SELF using a function named Value and __init__ , data and grad(Gradiant)
class Value:
    def __init__(self,data,_children=(),_op=""):
        self.data= data
        self.grad= 0.0
        self._prev=set(_children)
        self._op=_op
        self._backward=lambda:None
# Addition Function
    def __add__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        
        out=Value(self.data+other.data,(self,other),"+")
        # Backward Function 
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward=_backward
        return out
# A Function which Adds Value to a real number
    def __radd__(self,other):
        return self + other
# Multiply Function
    def __mul__(self,other):
        other= other if isinstance(other,Value) else Value(other)
        
        out= Value(self.data*other.data,(self,other),"*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward=_backward
        return out
# A Function which multiples Value to a real number
    def __rmul__(self,other):
        return self * other
# Representation Function
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
# Backward Function for a multi step equation
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
# Defining the negative in subtraction 
    def __neg__(self):
        return self * -1
# Subtraction Function
    def __sub__(self,other):
        return self+ (-other)
# Backward Subtraction Function
    def __rsub__(self,other):
        return other + (-self)
# Division (/) is Just Multiplication in Disguise!
# a/b is the exact same thing as a × b^(−1)
# Division Function aka the Power Function
    def __pow__(self, other):
        assert isinstance(other,(int,float))
        out= Value(self.data**other,(self,),f"**{other}")
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out
    def __truediv__(self, other):
        return self*(other**-1)
# Relu Funnction
    def relu(self):
        out= Value(0.0 if self.data<0 else self.data,(self,),"Relu")  
        def _backward():
            self.grad+=(out.data>0) * out.grad
        out._backward=_backward
        return out  
a= Value(5.0)
print("-"*100)
print("Data and Gradiant of A")
print(a.data)
print(a.grad)
b= Value(2.0)
# Data and Gradiant of C when A and B are Added
c=a+b
print("-"*100)
print("Data and Gradiant of C when A and B are Added")
print(c.data)
print(c.grad)
# Data and Gradiant of D when A and B are Multiplied
d=a*b
print("-"*100)
print("Data and Gradiant of D when A and B are Multiplied")
print(d.data)
print(d.grad)
# Parents of C and D
print("-"*100)
print("The Parents of C and D" )
print(c._prev)
print(d._prev)
# The oprators used in C and D
print("-"*100)
print("The oprators used in C and D")
print(c._op)
print(d._op)
# The Heart of Autograd — Local Derivatives
# Automatic Differentiation
print("-"*100)
print("Before Changing the Gradiant")
print(a.grad)
print(b.grad)
print("-"*100)
print("After Changing the Gradiant")
d.grad=2.0
d._backward()
print(a.grad)
print(b.grad)
print("-"*100)
print("The Grad of A,B,C after the multistep equation")
a = Value(2.0)
b = Value(3.0)
c = Value(10.0)
e = a * b      
L = e + c       
L.backward()    
print(a.grad)
print(b.grad)
print(c.grad)
# Make the Value function a Real number
X=Value(4.0)
Y=2*X+3
Y.backward()
print("-"*100)
print("The Grad of X will be")
print(X.grad)
a = Value(8.0)
b = Value(2.0)
print("-"*100)
# Subtraction:
y1 = a - b
y1.backward()
print("y1 = a - b:")
print("a.grad:", a.grad)  
print("b.grad:", b.grad)  
print("-"*100)
# Reset grads
a.grad = 0.0
b.grad = 0.0
print("-"*100)
# Division:
y2 = a / b
y2.backward()
print("y2 = a / b:")
print("a.grad:", a.grad)  
print("b.grad:", b.grad)  
print("-"*100)
# Testing ReLU
print("Testing ReLU:")
# Positive input
x1 = Value(5.0)
y1 = x1.relu()
y1.backward()
print("x1 (Positive) grad:", x1.grad)  
# Negative input
x2 = Value(-3.0)
y2 = x2.relu()
y2.backward()
print("x2 (Negative) grad:", x2.grad) 

"""Building a Neural Network Out of Value"""
#The Single Neuron

class Neuron:
    def __init__(self,nin,nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b=Value(random.uniform(-1,1))
        self.nonlin = nonlin
    def __call__(self,x):
        act= sum((wi*xi for wi , xi in zip(self.w,x)),self.b)
        return act.relu() if self.nonlin else act 
    def parameters(self):
        return self.w +[self.b]
print("-"*100)
print("Testing a Single Neuron:")

# Create a neuron that takes 2 inputs
n = Neuron(2)

# Give it an input vector x = [2.0, 3.0]
x = [Value(2.0), Value(3.0)]

# Fire the neuron!
out = n(x)
print("Neuron Output:", out)

# Run backward!
out.backward()
print("Weight Gradients:", [w.grad for w in n.w])
print("Bias Gradient:", n.b.grad)
# A Team of Neurons
class Layers:
    def __init__(self,nin,nout,**kwargs):
        self.neurons=[Neuron(nin,**kwargs) for _ in range(nout)]
    def __call__(self,x):
        outs =[n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs
    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params
print("-"*100)
print("Testing a Layer (2 inputs, 3 neurons):")

# Create a layer: 2 inputs -> 3 neurons
layer = Layers(2, 3)

# Pass the input vector x = [Value(2.0), Value(3.0)]
x = [Value(2.0), Value(3.0)]
layer_outs = layer(x)

print("Layer Outputs (3 neurons):", layer_outs)
print("Total Tunable Parameters in Layer:", len(layer.parameters()))
# Multi-Layer Perceptron
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layers(sz[i], sz[i+1], nonlin=(i != len(nouts) - 1)) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
print("-"*100)
print("Testing a Full Multi-Layer Perceptron (MLP):")

# Create a network: 2 inputs -> hidden layer of 4 -> hidden layer of 4 -> 1 output
model = MLP(2, [4, 4, 1])

x = [Value(2.0), Value(3.0)]
prediction = model(x)

print("Model Prediction:", prediction)
print("Total Tunable Parameters in MLP:", len(model.parameters()))

# Run full backprop from the prediction!
prediction.backward()
print("Grad of First Weight in Model:", model.parameters()[0].grad)
print("How many parameters got gradients:", len([p for p in model.parameters() if p.grad != 0.0]))
# Training the MLP
# 4 training examples (each with 2 features)
print("-"*100)
xs = [
    [2.0, 3.0],
    [3.0, -1.0],
    [0.5, 1.0],
    [1.0, 1.0]
]

# The target answers we want the network to predict:
ys = [1.0, -1.0, -1.0, 1.0]
for epoch in range(100):
    # 1. FORWARD PASS: Evaluate all inputs
    ypred = [model(x) for x in xs]

    # 2. COMPUTE LOSS: Mean Squared Error (how far off are we?)
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # 3. ZERO GRADIENTS: Reset the blame meters before each round
    for p in model.parameters():
        p.grad = 0.0

    # 4. BACKWARD PASS: One magic line!
    loss.backward()

    # 5. GRADIENT DESCENT UPDATE: Nudge each weight in the downhill direction
    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d} | Loss: {loss.data:.6f}")
print("-" * 100)
print("Targets:    ", ys)
print("Predictions:", [round(p.data, 4) for p in ypred])