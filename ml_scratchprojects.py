import numpy as np

#The "Crime Scene" of Linear Collapse (The Famous XOR Problem)
#The Linear Collapse(How the Loss stays the same without the AI actually learning anything)
input_dm=2
hidden_dm=4
output_dm=2
X=np.array([[0,0],[1,1],[1,0],[0,1]])
y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
print("-" * 100)
print("Shape of X and y_true")
print(X.shape)
print(y_true.shape)
print("-" * 100)
W1=np.random.randn(input_dm,hidden_dm)*np.sqrt(2.0 / input_dm)
W2=np.random.randn(hidden_dm,output_dm)*np.sqrt(2.0 / hidden_dm)
b1=np.zeros((1,hidden_dm))
b2=np.zeros((1,output_dm))
print("Shape of W1 W2 b1 and b2" )
print(W1.shape)
print(W2.shape)
print(b1.shape)
print(b2.shape)
def Affine_forward(X,W1,b1):
    Z=X @ W1 +b1
    return Z
#Adding the Relu back
def relu(Z):
     A=np.maximum(0,Z)
     return A
Z1=Affine_forward(X,W1,b1)
A1=relu(Z1)
Z2=Affine_forward(A1,W2,b2)
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
print("-" * 100)
print("--Current Loss--")
print(Loss)
print("-" * 100)
print("---Predicted and Original Data---")
print(y_hat)
print(y_true)
"""
The code when relu is not added-->
for traning in range(1000):
     #Forward Pass
     Z1=Affine_forward(X,W1,b1)
     Z2=Affine_forward(Z1,W2,b2)
     y_hat=softmax(Z2)
     #Compute Loss
     loss = Cross_Entropy_Loss(y_hat, y_true)
     #Backward Pass
     batch_size = y_true.shape[0]
     dZ2 = (y_hat - y_true) / batch_size
     dW2 = Z1.T @ dZ2
     db2 = np.sum(dZ2, axis=0, keepdims=True)
     dZ1 = dZ2 @ W2.T
     dW1 = X.T @ dZ1
     db1 = np.sum(dZ1, axis=0, keepdims=True)
     #Parameters
     lr=0.1
     W1-=lr*dW1
     W2-=lr*dW2
     b1-=lr*db1
     b2-=lr*db2
     #Print Progress every 100 steps 
     if traning%100==0:
          print(f"Traning{traning:2d} | Loss{loss:.6f}")"""
#The code when relu is added-->
for traning in range(1000):
     #Forward Pass
     Z1=Affine_forward(X,W1,b1)
     A1=relu(Z1)
     Z2=Affine_forward(A1,W2,b2)
     y_hat=softmax(Z2)
     #Compute Loss
     loss = Cross_Entropy_Loss(y_hat, y_true)
     #Backward Pass
     batch_size = y_true.shape[0]
     dZ2 = (y_hat - y_true) / batch_size
     dW2 = A1.T @ dZ2
     db2 = np.sum(dZ2, axis=0, keepdims=True)
     dA1=dZ2 @ W2.T
     dZ1 = dA1 *(Z1>0)
     dW1 = X.T @ dZ1
     db1 = np.sum(dZ1, axis=0, keepdims=True)
     #Parameters
     lr=0.1
     W1-=lr*dW1
     W2-=lr*dW2
     b1-=lr*db1
     b2-=lr*db2
     #Print Progress every 100 steps 
     if traning%100==0:
          print(f"Traning{traning:2d} | Loss{loss:.6f}")