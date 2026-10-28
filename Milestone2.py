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
        out=Value(self.data+other.data,(self,other),"+")
        # Backward Function 
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward=_backward
        return out
# Multiply Function
    def __mul__(self,other):
        out= Value(self.data*other.data,(self,other),"*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward=_backward
        return out
# Representation Function
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
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