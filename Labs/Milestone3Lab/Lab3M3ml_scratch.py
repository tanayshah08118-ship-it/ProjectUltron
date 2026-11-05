import numpy as np

# The Causal Mask & Entropy Collapse
text="hello"
T=len(text)
d_k=16
np.random.seed(42)
Q=np.random.randn(T,d_k)
K=np.random.randn(T,d_k)
scores=(Q@K.T)/np.sqrt(d_k)
print("-"*100)
print(scores)
def softmax(x):
    exp_x=np.exp(x-np.max(x,axis=-1,keepdims=True))
    return exp_x/np.sum(exp_x,axis=-1,keepdims=True)
atten_unmasked=softmax(scores)
print("-"*100)
print(np.round(atten_unmasked[0],3))
mask=np.tril(np.ones((T,T)))
scores_masked=scores.copy()
scores_masked[mask==0]=-np.inf
atten_masked=softmax(scores_masked)
print("-"*100)
print(np.round(atten_masked[0],3))
d_large=256
text_new="hello"
T_new=len(text_new)
Q_large=np.random.randn(T_new,d_large)
K_large=np.random.randn(T_new,d_large)
scores_unscaled=Q_large@K_large.T
scores_scaled=(Q_large@K_large.T)/np.sqrt(d_large)
prob_unscaled=softmax(scores_unscaled)
prob_scaled=softmax(scores_scaled)
print("-"*100)
print("Unscaled Softmax Row 0:\n", np.round(prob_unscaled[0], 4))
print("-"*100)
print("Scaled Softmax Row 0:\n", np.round(prob_scaled[0], 4))