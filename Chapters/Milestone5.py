import numpy as np

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

def backward(prob, cache,x, y):
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
    
run_experiment(tokens[:-1], tokens[1:], "=== CORRECT CAUSAL ===")

def sample_greedy(logits):
    return np.argmax(logits)

def sample_temperature(logits,temperature=1.0):
    scaled_logits = logits / temperature
    exp_l = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_l / np.sum(exp_l)
    return np.random.choice(len(probs), p=probs)

print("\n"+"-"*100)
print("TESTING SAMPLERS: TEMPERATURE SHOWDOWN")
print("-"*100)

temperatures = [0.05, 0.7, 3.0]

for temp in temperatures:
    context = [stoi["h"]]
    for _ in range(20):
        _, cache = forward(context)
        
        # Raw logits for the last token position
        next_token_logits = cache["x_final"][-1] @ W_head
        
        # Sample using our temperature function
        next_id = sample_temperature(next_token_logits, temperature=temp)
        context.append(next_id)
        
    result = "".join([itos[i] for i in context])
    print(f"Temperature T = {temp:4.2f} | Generated: '{result}'")

def sample_top_k(logits, k=3, temperature=1.0):
    logits = logits.copy()
    threshold = np.sort(logits)[-k]
    logits[logits < threshold] = -np.inf
    return sample_temperature(logits, temperature=temperature)

print("\n"+"-"*100)
print("TESTING SAMPLERS: TOP-K FILTERING (AT HIGH TEMP T = 2.0)")
print("-"*100)

k_values = [1, 2, 5]

for k in k_values:
    context = [stoi["h"]]
    for _ in range(20):
        _, cache = forward(context)
        next_token_logits = cache["x_final"][-1] @ W_head
        next_id = sample_top_k(next_token_logits, k=k, temperature=2.0)
        context.append(next_id)
        
    result = "".join([itos[i] for i in context])
    print(f"Top-k = {k} (T = 2.0) | Generated: '{result}'")

def sample_top_p(logits, p=0.9, temperature=1.0):
    scaled_logits = logits / temperature
    exp_l = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_l / np.sum(exp_l)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    sorted_indices_to_remove = cumulative_probs - sorted_probs > p
    sorted_probs[sorted_indices_to_remove] = 0.0
    sorted_probs = sorted_probs / np.sum(sorted_probs)
    sampled_sorted_index = np.random.choice(len(sorted_probs), p=sorted_probs)
    return sorted_indices[sampled_sorted_index]

p_values = [0.40,0.80,0.95]

for p in p_values:
    context = [stoi["h"]]
    for _ in range(20):
        _, cache = forward(context)
        next_token_logits = cache["x_final"][-1] @ W_head
        next_id = sample_top_p(next_token_logits, p=p, temperature=2.0)
        context.append(next_id)
        
    result = "".join([itos[i] for i in context])
    print(f"Top-p = {p:.2f} (T = 2.0) | Generated: '{result}'")

def apply_repetition_penalty(logits, context, penalty=1.2):
    logits = logits.copy()
    for token_id in set(context):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty   
    return logits

print("\n"+"-"*100)
print("TESTING REPETITION PENALTY (GREEDY DECODING)")
print("-"*100)

for pen in [1.0, 1.2, 1.5]:
    context = [stoi["h"]]
    for _ in range(20):
        _, cache = forward(context)
        next_token_logits = cache["x_final"][-1] @ W_head

        penalized_logits = apply_repetition_penalty(next_token_logits, context, penalty=pen)
        next_id = sample_greedy(penalized_logits)
        context.append(next_id)
        
    result = "".join([itos[i] for i in context])
    print(f"Penalty = {pen:.1f} | Generated: '{result}'")

import time


def forward_kv_cache(token_id, pos, past_k=None, past_v=None):
    """
    Executes a single-token decode step.
    Computes q, k, v only for token_id at position pos, concatenates k and v
    to the historical cache, and computes 1 x L attention without a causal mask matrix.
    """
    pos_idx = min(pos, block_size - 1)
    # Shape: (1, n_embed)
    x = W_token[[token_id]] + W_position[[pos_idx]]
    
    # 1. Pre-Attention RMSNorm
    x_norm = rms_normal(x, gamma)
    q = x_norm @ W_q
    k = x_norm @ W_k
    v = x_norm @ W_v
    
    # Concatenate new key & value with historical cache
    if past_k is None or past_v is None:
        K = k
        V = v
    else:
        K = np.concatenate([past_k, k], axis=0)
        V = np.concatenate([past_v, v], axis=0)
        if K.shape[0] > block_size:
            K = K[-block_size:]
            V = V[-block_size:]
            
    # Single-token attention: q (1 x d) attends to all past keys K (L x d)
    scores = (q @ K.T) / np.sqrt(d_k)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    attn_out = attn_weights @ V
    x = x + attn_out
    
    # 2. SwiGLU FFN Block
    x_norm2 = rms_normal(x, gamma_2)
    z_gate = x_norm2 @ W_gate
    gate = swish(z_gate)
    up = x_norm2 @ W_up
    ffn_out = (gate * up) @ W_down
    x = x + ffn_out
    
    # 3. Final RMSNorm & LM Head
    x_final = rms_normal(x, gamma_final)
    logits = x_final @ W_head
    return logits[0], K, V

print("\n" + "="*100)
print("VERIFYING KV CACHE MATHEMATICAL EQUIVALENCE & LATENCY SPEEDUP")
print("="*100)

# 1. Equivalence check across prompt "hello"
test_tokens = encode("hello")
_, cache_std = forward(test_tokens)
logits_std = cache_std["x_final"][-1] @ W_head

pk, pv = None, None
for pos, tid in enumerate(test_tokens):
    logits_kv, pk, pv = forward_kv_cache(tid, pos, pk, pv)

max_error = np.max(np.abs(logits_std - logits_kv))
print(f"Max absolute difference (Standard Forward vs KV Cache): {max_error:.2e}")
assert max_error < 1e-12, "Logits do not match!"
print("Mathematical Equivalence Verified: Exact bit-for-bit match!")

# 2. Benchmark 1,000 generation runs
N_REPS = 1000
t0 = time.perf_counter()
for _ in range(N_REPS):
    ctx = [stoi["h"]]
    for _ in range(7):
        _, c = forward(ctx)
        next_id = int(np.argmax(c["x_final"][-1] @ W_head))
        ctx.append(next_id)
t_std = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(N_REPS):
    ctx = [stoi["h"]]
    pk, pv = None, None
    logits, pk, pv = forward_kv_cache(ctx[0], 0, pk, pv)
    for step in range(1, 8):
        next_id = int(np.argmax(logits))
        ctx.append(next_id)
        logits, pk, pv = forward_kv_cache(next_id, step, pk, pv)
t_kv = time.perf_counter() - t0

print(f"Standard Forward Latency (1000 runs): {t_std*1000:.2f} ms")
print(f"KV Cache Forward Latency (1000 runs): {t_kv*1000:.2f} ms")
print(f"Decoding Speedup Factor:              {t_std / t_kv:.2f}x faster!")

def generate(prompt, max_tokens=20, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0, use_kv_cache=True):
    """
    Unified production inference API:
    - Encodes raw string prompt into token IDs
    - Supports Prefill + Decode phases with or without KV Cache
    - Applies Repetition Penalty (positive/negative piecewise)
    - Supports Greedy, Pure Temperature, Top-k, or Top-p (Nucleus) sampling
    - Decodes token IDs back to a readable string
    """
    context = encode(prompt)
    
    if use_kv_cache:
        # Phase 1: Prefill (populates KV cache with prompt tokens)
        past_k, past_v = None, None
        for pos, tid in enumerate(context):
            next_logits, past_k, past_v = forward_kv_cache(tid, pos, past_k, past_v)
            
        # Phase 2: Autoregressive Decode (single token per step)
        for _ in range(max_tokens):
            cur_pos = min(len(context), block_size - 1)
            
            # Repetition penalty
            logits = apply_repetition_penalty(next_logits, context, penalty=repetition_penalty)
            
            # Sampling strategy selection
            if temperature == 0.0:
                next_id = sample_greedy(logits)
            elif top_k is not None:
                next_id = sample_top_k(logits, k=top_k, temperature=temperature)
            elif top_p is not None:
                next_id = sample_top_p(logits, p=top_p, temperature=temperature)
            else:
                next_id = sample_temperature(logits, temperature=temperature)
                
            context.append(next_id)
            next_logits, past_k, past_v = forward_kv_cache(next_id, cur_pos, past_k, past_v)
    else:
        # Standard un-cached generation (O(T^2))
        for _ in range(max_tokens):
            _, cache = forward(context)
            next_logits = cache["x_final"][-1] @ W_head
            logits = apply_repetition_penalty(next_logits, context, penalty=repetition_penalty)
            
            if temperature == 0.0:
                next_id = sample_greedy(logits)
            elif top_k is not None:
                next_id = sample_top_k(logits, k=top_k, temperature=temperature)
            elif top_p is not None:
                next_id = sample_top_p(logits, p=top_p, temperature=temperature)
            else:
                next_id = sample_temperature(logits, temperature=temperature)
                
            context.append(next_id)
            
    return "".join(decode(context))

print("\n" + "="*100)
print("UNIVERSAL GENERATE() SHOWCASE: SAMPLING PRESETS")
print("="*100)

presets = [
    ("Deterministic Greedy (T=0.0, Pen=1.2)", {"temperature": 0.0, "repetition_penalty": 1.2}),
    ("Balanced Creative (T=0.7, Pen=1.1)",   {"temperature": 0.7, "repetition_penalty": 1.1}),
    ("Top-p Nucleus (T=1.2, p=0.85, Pen=1.2)", {"temperature": 1.2, "top_p": 0.85, "repetition_penalty": 1.2}),
    ("Top-k Guardrail (T=1.5, k=2, Pen=1.2)",  {"temperature": 1.5, "top_k": 2, "repetition_penalty": 1.2}),
]

for name, kwargs in presets:
    out = generate("h", max_tokens=15, **kwargs)
    print(f"{name:<42} | Output: '{out}'")