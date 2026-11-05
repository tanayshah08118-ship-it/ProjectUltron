import numpy as np

input_dm=2
hidden_dm=4
output_dm=2
b1=np.zeros((1,hidden_dm))
b2=np.zeros((1,output_dm))
W1 = np.random.randn(input_dm, hidden_dm) * np.sqrt(2.0 / input_dm)
W2 = np.random.randn(hidden_dm, output_dm) * np.sqrt(2.0 / hidden_dm)
X = np.random.randn(3,input_dm)
W = np.random.randn(2,hidden_dm) * np.sqrt(2.0 / input_dm)
b = np.zeros((1,hidden_dm))
def Affine_forward(X,W,b):
   Z= X @ W +b
   return Z
Z1= Affine_forward(X,W,b)
def relu(Z):
    A1=np.maximum(0,Z)
    return A1
A1=relu(Z1)
X = np.random.randn(3,input_dm)
Z1= Affine_forward(X,W1,b1)
def relu(Z):
    A1=np.maximum(0,Z)
    return A1
A1= relu(Z1)
Z2= Affine_forward(A1,W2,b2)
def softmax(Z):
    exp_shifted=Z-np.max(Z, axis=1,keepdims=True)
    exps=np.exp(exp_shifted)
    return exps / np.sum(exps, axis=1,keepdims=True)
y_hat= softmax(Z2)
y_true=np.array([[1,0],[0,1],[1,0]])
def cross_entropy_loss(y_pred,y_true):
    eps=1e-15
    loss= -np.sum(y_true * np.log(y_pred + eps)) / y_true.shape[0]
    return loss
loss=cross_entropy_loss(y_hat,y_true)
batch_size=y_true.shape[0]
dZ2=(y_hat-y_true) / batch_size
dW2= A1.T @ dZ2
db2=np.sum(dZ2, axis=0, keepdims=True)
dA1 = dZ2 @ W2.T
dZ1 = dA1 * (Z1 > 0)
dW1 = X.T @ dZ1
db1 = np.sum(dZ1, axis=0, keepdims=True)
#Learning Rate(lr)-->
# lr=0.00001  <--Condition A: The Snail
# lr=0.1   <--Condition B: The Goldilocks
lr=100
W1-=lr*dW1
b1-=lr*db1
W2-=lr*dW2
b2-=lr*db2
for training in range(100):
  # Forward Pass (indented by 4 spaces)
  Z1 = Affine_forward(X, W1, b1)
  A1 = relu(Z1)
  Z2 = Affine_forward(A1, W2, b2)
  y_hat = softmax(Z2)
# Compute Loss
  loss = cross_entropy_loss(y_hat, y_true)
# Backward Pass
  batch_size = y_true.shape[0]
  dZ2 = (y_hat - y_true) / batch_size
  dW2 = A1.T @ dZ2
  dA1 = dZ2 @ W2.T
  db2 = np.sum(dZ2, axis=0, keepdims=True)
  dZ1 = dA1 * (Z1 > 0)
  dW1 = X.T @ dZ1
  db1 = np.sum(dZ1, axis=0, keepdims=True)
# Update Weights
  W1 -= lr * dW1
  b1 -= lr * db1
  W2 -= lr * dW2
  b2 -= lr * db2
# Print progress every 10 steps
  if training % 10 == 0:
    print(f"Traning {training:2d} | Loss: {loss:.6f}")
