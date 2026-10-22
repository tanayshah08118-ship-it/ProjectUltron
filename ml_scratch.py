import numpy as np

"""
Linera Algebra-->
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
"""
Affine  Transformation-->
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
"""
Non-Linear Activation Function (relu)-->
def relu(Z):
    A1=np.maximum(0,Z)
    return A1
A1=relu(Z1)
print("---Before relu---")
print(Z1)
print("---After relu---")
print(A1)
"""
"""
Step1} Passing through Layer2-->
input_dm=2
hidden_dm=4
output_dm=2
b1 = np.zeros((1,hidden_dm))
b2 = np.zeros((1,output_dm))
X = np.random.randn(3,input_dm)
W1 = np.random.randn(input_dm,hidden_dm) * np.sqrt(1 / input_dm)
W2 = np.random.randn(hidden_dm,output_dm) * np.sqrt(1 / hidden_dm)
def Affine_forward(X,W,b):
    Z= X @ W + b
    return Z
Z1= Affine_forward(X,W1,b1)
def relu(Z):
    A1=np.maximum(0,Z)
    return A1
A1= relu(Z1)
Z2= Affine_forward(A1,W2,b2)
"""
"""
Step2} The "softmax" Function (aka the Probability Function)-->
def softmax(Z):
    exp_shifted=Z-np.max(Z, axis=1,keepdims=True)
    exps=np.exp(exp_shifted)
    return exps / np.sum(exps, axis=1,keepdims=True)
y_hat= softmax(Z2)
print("---Final pridicted posibility are---")
print(y_hat)
"""
