import numpy as np

#The "Lie Detector" (Softmax & Cross-Entropy Dynamics)
y_true=np.array([[1,0]])
Z1=np.array([[10,-10]])   # <--Z1 When the model is 100% acurate (Confident & Correct)
#Z1=np.array([[1.0,1.0]])  # <--Z1 When the model is 50% acurate (Completely Unsure (50/50))
#Z1=np.array([[-10.0,10.0]]) # <--Z1 When the model is 0% acurate (Arrogant & Wrong)
def softmax(Z):
     exp_shifted=Z-np.max(Z, axis=1,keepdims=True)
     exps=np.exp(exp_shifted)
     return exps / np.sum(exps, axis=1,keepdims=True)
y_hat= softmax(Z1)
def Cross_entropy_loss(y_pred,y_true):
     eps=1e-15
     loss= -np.sum(y_true * np.log(y_pred + eps)) / y_true.shape[0]
     return loss
Loss=Cross_entropy_loss(y_hat,y_true)
print("--- Scenario 1: Confident and Correct ---")
print("Probabilities (y_hat):", y_hat)
print("Loss Penalty:", Loss)