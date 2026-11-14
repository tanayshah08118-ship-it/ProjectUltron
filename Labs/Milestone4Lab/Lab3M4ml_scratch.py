import numpy as np

# Setup-->
text="hello"
char=sorted(list(set(text)))
v_size=len(char)
stoi={ch:i for i,ch in enumerate(char)}
itos={i:ch for i,ch in enumerate(char)}

def encode(s):
    return [stoi[c]for c in s]
def decode(l):
    return [itos[i]for i in l]

tokens=encode("hello")
x=tokens[:-1]
y=tokens[1:]
np.random.seed(42)

d_k=16
n_embed=16
d_ffn=64
block_size=8
gamma=np.ones(n_embed)
gamma_2=np.ones(n_embed)
gamma_final=np.ones(n_embed)
W_token = np.random.randn(v_size, n_embed) * 0.1
W_position = np.random.randn(block_size, n_embed) * 0.1
W_q = np.random.randn(n_embed, n_embed) * 0.1
W_k = np.random.randn(n_embed, n_embed) * 0.1
W_v = np.random.randn(n_embed, n_embed) * 0.1
W_gate = np.random.randn(n_embed, d_ffn) * 0.1
W_up = np.random.randn(n_embed, d_ffn) * 0.1
W_down = np.random.randn(d_ffn, n_embed) * 0.1
W_head = np.random.randn(n_embed, v_size) * 0.01

def rms_normal(x,gamma,eps=1e-5):
    rms=np.sqrt(np.mean(x**2,axis=-1,keepdims=True)+eps)
    return (x/rms)*gamma

def swish(z):
    return z/(1.0+np.exp(-z))

def forward(token_ids):
    token_ids=token_ids[-block_size:]
    T=len(token_ids)
    x=W_token[token_ids]+W_position[:T]
    x_normal=rms_normal(x,gamma)
    Q=x_normal@W_q
    K=x_normal@W_k
    V=x_normal@W_v
    scores=(Q@K.T)/np.sqrt(d_k)
    mask=np.tril(np.ones((T,T)))
    scores[mask==0]=-np.inf
    exp_scores=np.exp(scores-np.max(scores,axis=-1,keepdims=True))
    attn_weights=exp_scores/np.sum(exp_scores,axis=-1,keepdims=True)
    attn_out=attn_weights@V
    x+=attn_out
    x_normal2=rms_normal(x,gamma_2)
    z_gate=x_normal2@W_gate
    gate=swish(z_gate)
    up=x_normal2@W_up
    hidden=gate*up
    ffn_out=hidden@W_down
    x+=ffn_out
    x_final=rms_normal(x,gamma_final)
    logits=x_final@W_head
    exp_logits=np.exp(logits-np.max(logits,axis=-1,keepdims=True))
    probability=exp_logits/np.sum(exp_logits,axis=-1,keepdims=True)
    cache={"x_final":x_final,"x":x ,"attn_out":attn_out,"x_normal2":x_normal2,"gate":gate,"up":up,"hidden":hidden,"ffn_out":ffn_out,"z_gate":z_gate,"x_normal":x_normal,"attn_weights":attn_weights,"Q":Q,"K":K,"V":V}
    return probability,cache
prob,cache=forward(x)
correct_prob=prob[range(len(y)),y]
Loss=-np.mean(np.log(correct_prob))

dlogits=prob.copy()
dlogits[range(len(y)),y] -= 1.0
dlogits/=len(y)
x_final=cache["x_final"]
dW_head=x_final.T@dlogits
dx_final=dlogits@W_head.T
new_prob,_=forward(x)
new_loss=-np.mean(np.log(new_prob[range(len(y)),y]))
dx_final=dlogits@W_head.T 
dffn_out=dx_final.copy()
hidden=cache["hidden"]
dW_down=hidden.T@dffn_out
dhidden=dffn_out@W_down.T
dup=dhidden*cache["gate"]
dgate=dhidden*cache["up"]
x_normal2=cache["x_normal2"]
dW_up=x_normal2.T@dup

def dswish(z):
    return 1.0/(1.0+np.exp(-z))

z_gate=cache["z_gate"]
s=dswish(z_gate)
swish_grad=s+cache["gate"]*(1.0-s)
dz_gate=dgate*swish_grad
dW_gate=x_normal2.T@dz_gate
dx_normal2 = dz_gate @ W_gate.T + dup @ W_up.T
dx = dx_final + dx_normal2
dattn_out = dx.copy()
dV = cache["attn_weights"].T @ dattn_out
x_normal = cache["x_normal"]
dW_v = x_normal.T @ dV
dattn_weights = dattn_out @ cache["V"].T
sum_grad = np.sum(dattn_weights * cache["attn_weights"], axis=-1, keepdims=True)
dscores = cache["attn_weights"] * (dattn_weights - sum_grad)
T = len(x)
mask = np.tril(np.ones((T, T)))
dscores[mask == 0] = 0.0
dscores = dscores / np.sqrt(d_k)
dQ = dscores @ cache["K"]
dK = dscores.T @ cache["Q"]
dW_q = x_normal.T @ dQ
dW_k = x_normal.T @ dK
dx_normal = dQ @ W_q.T + dK @ W_k.T + dV @ W_v.T
dx_embed = dx + dx_normal
dW_position = np.zeros_like(W_position)
dW_position[:T] += dx_embed
dW_token = np.zeros_like(W_token)
np.add.at(dW_token, x, dx_embed)

def backward(prob, cache, y):
    dlogits=prob.copy()
    dlogits[range(len(y)),y] -= 1.0
    dlogits/=len(y)
    x_final=cache["x_final"]
    dW_head=x_final.T@dlogits
    dx_final=dlogits@W_head.T
    dffn_out=dx_final.copy()
    hidden=cache["hidden"]
    dW_down=hidden.T@dffn_out
    dhidden=dffn_out@W_down.T
    dup=dhidden*cache["gate"]
    dgate=dhidden*cache["up"]
    x_normal2=cache["x_normal2"]
    dW_up=x_normal2.T@dup
    def dswish(z):
        return 1.0/(1.0+np.exp(-z))
    z_gate=cache["z_gate"]
    s=dswish(z_gate)
    swish_grad=s+cache["gate"]*(1.0-s)
    dz_gate=dgate*swish_grad
    dW_gate=x_normal2.T@dz_gate
    dx_normal2 = dz_gate @ W_gate.T + dup @ W_up.T
    dx = dx_final + dx_normal2
    dattn_out = dx.copy()
    dV = cache["attn_weights"].T @ dattn_out
    x_normal = cache["x_normal"]
    dW_v = x_normal.T @ dV
    dattn_weights = dattn_out @ cache["V"].T
    sum_grad = np.sum(dattn_weights * cache ["attn_weights"], axis=-1, keepdims=True)
    dscores = cache["attn_weights"] * (dattn_weights -  sum_grad)
    T = len(x)
    mask = np.tril(np.ones((T, T)))
    dscores[mask == 0] = 0.0
    dscores = dscores / np.sqrt(d_k)
    dQ = dscores @ cache["K"]
    dK = dscores.T @ cache["Q"]
    dW_q = x_normal.T @ dQ
    dW_k = x_normal.T @ dK
    dx_normal = dQ @ W_q.T + dK @ W_k.T + dV @ W_v.T
    dx_embed = dx + dx_normal
    dW_position = np.zeros_like(W_position)
    dW_position[:T] += dx_embed
    dW_token = np.zeros_like(W_token)
    np.add.at(dW_token, x, dx_embed)
    return dW_head, dW_down, dW_up, dW_gate, dW_q, dW_k, dW_v, dW_position, dW_token

# Experiment 1: The 9-Gradient Symmetry Check-->
print("-"*100)
print("EXPERIMENT 1: THE 9-GRADIENT SHAPE SYMMETRY CHECK")
print("-"*100)

weights = [W_head, W_down, W_up, W_gate, W_q, W_k, W_v, W_position, W_token]
names = ["W_head", "W_down", "W_up", "W_gate", "W_q", "W_k", "W_v", "W_pos", "W_token"]
grads = [dW_head, dW_down, dW_up, dW_gate, dW_q, dW_k, dW_v, dW_position, dW_token]

for name, W, dW in zip(names, weights, grads):
    match = (W.shape == dW.shape)
    print(f"{name:10s} Weight: {str(W.shape):10s} | Grad: {str(dW.shape):10s} | Match: {match}")


# Experiment 2: The Duplicate Token Accumulation Trap-->
print("-"*100)
print("EXPERIMENT 2: THE DUPLICATE TOKEN ACCUMULATION TRAP")
print("-"*100)

l_id = stoi['l']
print(f"Input x: {x}")
print(f"Token ID for 'l': {l_id} (appears at position 2 and position 3)\n")

# Method A: Naive slice addition (The Trap!)
dW_token_naive = np.zeros_like(W_token)
dW_token_naive[x] += dx_embed

# Method B: Atomic accumulation (np.add.at)
dW_token_correct = np.zeros_like(W_token)
np.add.at(dW_token_correct, x, dx_embed)

# Compare the gradients stored for 'l'
grad_naive = dW_token_naive[l_id]
grad_correct = dW_token_correct[l_id]

print(f"Naive dW_token['l'] norm (+=) :     {np.linalg.norm(grad_naive):.4f}")
print(f"Correct dW_token['l'] norm (add.at): {np.linalg.norm(grad_correct):.4f}")

# Did naive += simply overwrite position 2 with position 3?
overwritten = np.allclose(grad_naive, dx_embed[3])
print(f"Did naive slice overwrite pos 2 with pos 3? {overwritten}")