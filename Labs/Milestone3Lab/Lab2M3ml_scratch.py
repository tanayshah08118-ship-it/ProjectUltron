import numpy as np

def rms_normal(x,gamma,eps=1e-5):
    rms=np.sqrt(np.mean(x**2,axis=-1,keepdims=True)+eps)
    return (x/rms)*gamma

# Experiment 1 (The Un-normalized Explosion)
d=16
np.random.seed(42)
x=np.random.randn(1,d)
print("-"*100)
print("Before the RMSNorms Regulation")
for Layers in range(1,11):
    W=np.random.randn(d,d)
    x=x@W
    rms_val=np.sqrt(np.mean(x**2,axis=-1,keepdims=True))
    print(f"Layer {Layers}: RMS = {rms_val[0, 0]:.4f}")

# Experiment 2: The RMSNorm Regulator
np.random.seed(42)
x_normal=np.random.randn(1,d)
gamma=np.ones(d)
print("-"*100)
print("After the RMSNorms Regulation")
for Layers in range(1,11):
    x_normal=rms_normal(x_normal,gamma)
    W=np.random.randn(d,d)
    x_normal=x_normal@W
    rms_val=np.sqrt(np.mean(x_normal**2,axis=-1,keepdims=True))
    print(f"Layer{Layers}:RMS={rms_val[0,0]:.4f}")

# Experiment 3: Why Did Modern LLMs Abandon Classic LayerNorm?
def layer_norm(x,gamma,beta,eps=1e-5):
    mean=np.mean(x,axis=-1,keepdims=True)
    variance=np.var(x,axis=-1,keepdims=True)
    return ((x-mean)/np.sqrt(variance+eps))*gamma+beta
test_array=np.array([[1.0,2.0,3.0,4.0,5.0]])
gamma=np.ones(5)
beta=np.zeros(5)
out_ln=layer_norm(test_array,gamma,beta)
out_rms=rms_normal(test_array,gamma)
print("-"*100)
print("LayerNorm Output:\n",out_ln)
print("-"*100)
print("RMSNorm Output:\n",out_rms)
ln_mean=np.mean(out_ln)
rms_mean=np.mean(out_rms)
print("-"*100)
print("Mean of Out Layer :\n",ln_mean)
print("-"*100)
print("Mean of Out RMS:\n",rms_mean)