import numpy as np

# How an LLM Reads Training Data (Causal Next-Token Slicing)
text="hello"
char=sorted(list(set(text)))
v_size=len(char)
stoi={ch:i for i,ch in enumerate(char)}
itos={i:ch for i,ch in enumerate(char)}
def encode(s):
    return [stoi[c]for c in s]
def decode(l):
    return [itos[i]for i in l]
tokens=encode(text)
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
def forward(tokens_id):
    tokens_id=tokens_id[-block_size:]
    T=len(tokens_id)
    x=W_token[tokens_id]+W_position[:T]
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
def backward(prob,cache,x,y):
    dlogits=prob.copy()
    dlogits[range(len(y)),y]-=1.0
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
    dx_normal2=dz_gate@W_gate.T+dup@W_up.T
    dx=dx_final+dx_normal2
    dattn_out=dx.copy()
    dV=cache["attn_weights"].T@dattn_out
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

lr=0.05
def run_experiment(x,y,label):
    global W_token, W_position, W_q, W_k, W_v, W_gate, W_up, W_down, W_head
    np.random.seed(42)
    W_token = np.random.randn(v_size, n_embed) * 0.1
    W_position = np.random.randn(block_size, n_embed) * 0.1
    W_q = np.random.randn(n_embed, n_embed) * 0.1
    W_k = np.random.randn(n_embed, n_embed) * 0.1
    W_v = np.random.randn(n_embed, n_embed) * 0.1
    W_gate = np.random.randn(n_embed, d_ffn) * 0.1
    W_up = np.random.randn(n_embed, d_ffn) * 0.1
    W_down = np.random.randn(d_ffn, n_embed) * 0.1
    W_head = np.random.randn(n_embed, v_size) * 0.01
    for steps in range(100):
        prob,cache=forward(x)

        loss=-np.mean(np.log(prob[range(len(y)),y]))

        dW_head, dW_down, dW_up, dW_gate, dW_q, dW_k, dW_v, dW_position, dW_token=backward(prob,cache,x,y)

        W_head -= lr * dW_head
        W_down -= lr * dW_down
        W_up -= lr * dW_up
        W_gate -= lr * dW_gate
        W_q -= lr * dW_q
        W_k -= lr * dW_k
        W_v -= lr * dW_v
        W_position -= lr * dW_position
        W_token -= lr * dW_token

        if steps% 10 == 0 or steps == 99:
            print(f"Step {steps:02d} | Loss: {loss:.4f}")

    context = [stoi["h"]]
    for _ in range(4):
        prob, _ = forward(context)
        next_id = np.argmax(prob[-1])  
        context.append(next_id)

    generated_text = "".join([itos[i] for i in context])
    print("Prompt:     'h'")
    print("Generated: ", generated_text)
    
run_experiment(tokens[:-1], tokens[1:], "=== EXPERIMENT A: CORRECT CAUSAL ===")
run_experiment(tokens[1:], tokens[:-1], "=== EXPERIMENT B: BROKEN ANTI-CAUSAL ===")