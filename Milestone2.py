"""Building Your Own Autograd Engine"""

#Initilazing the CLASS and SELF using a function named Value and __init__ , data and grad(Gradiant)
class Value:
    def __init__(self,data,_children=(),_op=""):
        self.data= data
        self.grad= 0.0
        self._prev=set(_children)
        self._op=_op
# Add Function
    def __add__(self, other):
        out=Value(self.data+other.data,(self,other),"+")
        return out
# Multiply Function
    def __mul__(self,other):
        out= Value(self.data*other.data,(self,other),"*")
        return out
# Representation Function
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
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