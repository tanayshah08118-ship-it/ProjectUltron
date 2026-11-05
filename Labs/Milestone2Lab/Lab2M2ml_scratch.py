# The Graph Amnesia Test (Severing the Ancestry Chain)!
class Value:
    def __init__(self,data,_children=(),_op=""):
        self.data=data
        self.grad=0.0
        self._prev=set(_children)
        self._op=_op
        self._backward= lambda: None
    def __add__(self,other):
        other=other if isinstance(other,Value)else Value(other)
        out = Value(self.data+other.data,(self,other),_op="+") # <-- if the (self,other) are removed it causes THE WEIGHTS TO NEVER UPDATE
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
        other = other if isinstance(other,Value)else Value(other)
        out=Value(self.data*other.data,(self,other),_op="*") # <-- if the (self,other) are removed it causes THE WEIGHTS TO NEVER UPDATE
        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backward
        return out
a= Value(2.0)
b= Value(3.0)
c= Value(10.0)
d=a*b
L=d+c
L.backward()
print("-"*100)
print("Gradiant of A,B,C and D :")
print(a.grad)
print(b.grad)
print(c.grad)
print(d.grad)