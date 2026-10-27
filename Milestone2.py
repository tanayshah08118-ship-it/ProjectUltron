"""Building Your Own Autograd Engine"""

#Initilazing the CLASS and SELF using a function named Value and __init__ , data and grad(Gradiant)
class Value:
    def __init__(self,data):
        self.data= data
        self.grad= 0.0
    def __add__(self, other):
        out=Value(self.data+other.data)
        return out
    def __mul__(self,other):
        out= Value(self.data*other.data)
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