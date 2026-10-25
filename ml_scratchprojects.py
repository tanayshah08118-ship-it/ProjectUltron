import numpy as np

#The "Crime Scene" of Linear Collapse (The Famous XOR Problem)
input_dm=2
hidden_dm=4
output_dm=2
X=np.array([[0,0],[1,1],[1,0],[0,1]])
y_true=np.array([[0,0],[0,1],[1,0],[1,1]])
print("Shape of X and y_true")
print(X.shape)
print(y_true.shape)
print("-" * 50)
W1=np.random.randn(input_dm,hidden_dm)*np.sqrt(2.0 / input_dm)
W2=np.random.randn(hidden_dm,output_dm)*np.sqrt(2.0 / hidden_dm)
b1=np.zeros((1,hidden_dm))
b2=np.zeros((1,output_dm))
print("Shape of W1 W2 b1 and b2" )
print(W1.shape)
print(W2.shape)
print(b1.shape)
print(b2.shape)
X = np.random.randn(4,input_dm)
W = np.random.randn(2,hidden_dm) * np.sqrt(2.0 / input_dm)
b = np.zeros((1,hidden_dm))
def Affine_forward(X,W,b):
    Z=X @ W +b
    return Z
Z1=Affine_forward(X,W,b)
A1=np.maximum(0,Z1)
Z2=Affine_forward(Z1,W2,b2)
def softmax(Z):
     exp_shifted=Z-np.max(Z, axis=1,keepdims=True)
     exps=np.exp(exp_shifted)
     return exps / np.sum(exps, axis=1,keepdims=True)
y_hat= softmax(Z2)
def Cross_Entropy_Loss(y_pred,y_true):
     eps=1e-15
     loss= -np.sum(y_true * np.log(y_pred + eps)) / y_true.shape[0]
     return loss
Loss=Cross_Entropy_Loss(y_hat,y_true)
print("--Current Loss--")
print(Loss)
print("---Predicted and Original Data---")
print(y_hat)
print(y_true)
