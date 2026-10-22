import numpy as np

"""
Linera Algebra-->"""
input_dm=2
hidden_dm=4
output_dm=2
b1=np.zeros((1,hidden_dm))
b2=np.zeros((1,output_dm))
W1 = np.random.randn(input_dm, hidden_dm) * np.sqrt(2.0 / input_dm)
W2 = np.random.randn(hidden_dm, output_dm) * np.sqrt(2.0 / hidden_dm)
print(W1.shape)
print(W2.shape)

"""
Affine  Transformation-->"""
input_dm=2
hidden_dm=4 
X = np.random.randn(3,input_dm)
W = np.random.randn(2,hidden_dm) * np.sqrt(2.0 / input_dm)
b = np.zeros((1,hidden_dm))
def Affine_forward(X,W,b):
   Z= X @ W +b
   return Z
Z1= Affine_forward(X,W,b)
print("Z1 shape:", Z1.shape)

"""
Non-Linear Activation Function (relu)-->"""
def relu(Z):
    A1=np.maximum(0,Z)
    return A1
A1=relu(Z1)
print("---Before relu---")
print(Z1)
print("---After relu---")
print(A1)
"""
Step1} Passing through Layer2-->
"""
