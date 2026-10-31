# The Accumulation Trap (Why += is Mandatory)
class Value:
    def __init__(self,data,_children=(),_op=""):
        self.data=data
        self.grad=0.0
        self._prev=set(_children)
        self._op=_op
        self._backward=lambda:None
# Testing the Addition Branch
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out=Value(self.data+other.data,(self,other),"+")
        def _backward():
            self.grad+=1.0*out.grad
            other.grad+=1.0*out.grad
        out._backward=_backward
        return out
# Important Function
# Backward pass for Multi step Function
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
    def __mul__(self, other):
        other = other if isinstance(other,Value)else Value(other)
        out=Value(self.data*other.data,(self,other),"*")
        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backward
        return out
x=Value(3.0)
'''y=x+x+x
y.backward()
print("-"*100)
print("Gradiant of x is",x.grad)'''
y=x*x
y.backward()
print("-"*100)
print("Gradiant of x is",x.grad)