import numpy as np

# Stage 1: How an LLM Reads Training Data
token=[2,1,5,5,7]
x=token[:-1] # <--The Input Data
y=token[1:]  # <--The Target Data

# Stage 2: The Cross-Entropy Loss (Measuring Surprise)
# L=−logP (correct token)
# The Legendary Sanity Check
# L_initial​=−log(V1​)=log(V)=log(11)~=2.4 

# Stage 3: Building the Training Loop
# Set Up for Cross Entropy Loss Function
text="hello tokenizer"
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
W_head = np.random.randn(n_embed, v_size) * 0.01 # <--Adding the Weight 
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
print("-"*100)
print(f"Loss:{Loss}")

# Stage 4: The Backward Pass — How the Transformer Learns
dlogits=prob.copy()
dlogits[range(len(y)),y] -= 1.0
dlogits/=len(y)
print("-"*100)
print("dlogits Shape:\n",dlogits.shape)
x_final=cache["x_final"]
dW_head=x_final.T@dlogits
dx_final=dlogits@W_head.T
print("-" * 100)
print("dW_head Shape:", dW_head.shape)
print("dW_head Sample:\n", np.round(dW_head[:3, :3], 4))
new_prob,_=forward(x)
new_loss=-np.mean(np.log(new_prob[range(len(y)),y]))
print(new_loss)

# Stage 5: Flowing Gradients Backward into the Backbone
dx_final=dlogits@W_head.T 
# Step 1 of the Backbone: The Residual Addition Rule
# Step 2: Backprop through SwiGLU (W_down)
# Step 3: Saving SwiGLU Tensors in the Forward Cache
# Step 4: Computing SwiGLU Gradients
dffn_out=dx_final.copy()
hidden=cache["hidden"]
dW_down=hidden.T@dffn_out
dhidden=dffn_out@W_down.T
dup=dhidden*cache["gate"]
dgate=dhidden*cache["up"]
x_normal2=cache["x_normal2"]
dW_up=x_normal2.T@dup
print(dW_down.shape,dW_up.shape)
def dswish(z):
    return 1.0/(1.0+np.exp(-z))
z_gate=cache["z_gate"]
s=dswish(z_gate)
swish_grad=s+cache["gate"]*(1.0-s)
dz_gate=dgate*swish_grad
dW_gate=x_normal2.T@dz_gate
print("-"*100)
print(dW_gate)
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
print("-" * 100)
print("ALL 9 TRANSFORMER GRADIENTS COMPUTED:")
print("dW_head:", dW_head.shape)
print("dW_down:", dW_down.shape)
print("dW_up:", dW_up.shape)
print("dW_gate:", dW_gate.shape)
print("dW_q:", dW_q.shape)
print("dW_k:", dW_k.shape)
print("dW_v:", dW_v.shape)
print("dW_pos:", dW_position.shape)
print("dW_token:", dW_token.shape)

# The Backward Propagation
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
lr = 0.05

print("-" * 50)
print("TRAINING THE TRANSFORMER...")
print("-" * 50)

for step in range(100):
    # 1. Forward Pass
    prob, cache = forward(x)
    
    # 2. Compute Loss
    loss = -np.mean(np.log(prob[range(len(y)), y]))
    
    # 3. Backward Pass
    dW_head, dW_down, dW_up, dW_gate, dW_q, dW_k, dW_v, dW_position, dW_token = backward(prob, cache, y)
    
    # 4. Gradient Descent Update (W -= lr * dW)
    W_head -= lr * dW_head
    W_down -= lr * dW_down
    W_up -= lr * dW_up
    W_gate -= lr * dW_gate
    W_q -= lr * dW_q
    W_k -= lr * dW_k
    W_v -= lr * dW_v
    W_position -= lr * dW_position
    W_token -= lr * dW_token
    
    # Print progress every 10 steps
    if step % 10 == 0 or step == 99:
        print(f"Step {step:02d} | Loss: {loss:.4f}")

# Test Autoregressive Generation with Trained Weights
print("\n" + "=" * 50)
print("AUTOREGRESSIVE GENERATION TEST:")
print("=" * 50)

context = [stoi["h"]]
for _ in range(100):
    prob, _ = forward(context)
    next_id = np.argmax(prob[-1])  # Greedy pick highest probability
    context.append(next_id)

generated_text = "".join([itos[i] for i in context])
print("Prompt:     'h'")
print("Generated: ", generated_text)