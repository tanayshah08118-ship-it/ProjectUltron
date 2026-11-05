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
# Compute the Theoretical Benchmark
theory_loss= np.log(v_size)
print(f"Theoretical Humble Loss: ln({v_size}) = {theory_loss:.4f}")
# Set Up the Model Backbone
np.random.seed(42)
n_embed=16
d_k=16
d_ffn=64
block_size=8
gamma=np.ones(n_embed)
gamma_2=np.ones(n_embed)
gamma_final=np.ones(n_embed)
W_token=np.random.randn(v_size,n_embed)*0.1
W_pos=np.random.randn(block_size,n_embed)*0.1
W_q=np.random.randn(n_embed,n_embed)*0.1
W_k=np.random.randn(n_embed,n_embed)*0.1
W_v=np.random.randn(n_embed,n_embed)*0.1
W_gate=np.random.randn(n_embed,d_ffn)*0.1
W_up=np.random.randn(n_embed,d_ffn)*0.1
W_down=np.random.randn(d_ffn,n_embed)*0.1
W_head=np.random.randn(n_embed,v_size)
def rms_normal(x,gamma,eps=1e-5):
    rms=np.sqrt(np.mean(x**2,axis=-1,keepdims=True)+eps)
    return (x/rms)*gamma
def swish(z):
    return z/(1.0+np.exp(-z))
def forward(tokens_id):
    tokens_id=tokens_id[-block_size:]
    T=len(tokens_id)
    x=W_token[tokens_id]+W_pos[:T]
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
x_final = cache["x_final"]

# Condition A: The Overconfident Trap (σ=1.0)
np.random.seed(42)
W_head_naive = np.random.randn(n_embed, v_size)  # std = 1.0

# 1. Logits & Softmax
logits_naive = x_final @ W_head_naive
exp_naive = np.exp(logits_naive - np.max(logits_naive, axis=-1, keepdims=True))
prob_naive = exp_naive / np.sum(exp_naive, axis=-1, keepdims=True)

# 2. Correct token probabilities & Loss
correct_p_naive = prob_naive[range(len(y)), y]
loss_naive = -np.mean(np.log(correct_p_naive))

print("-" * 60)
print("--- CONDITION A: UN-SCALED W_head (std = 1.0) ---")
print(f"Logit Range:           [{np.min(logits_naive):.2f}, {np.max(logits_naive):.2f}]")
print(f"Logit Dynamic Spread:  {np.ptp(logits_naive):.2f}")
print(f"P(correct tokens):     {np.round(correct_p_naive, 4)}")
print(f"Initial Loss:          {loss_naive:.4f}")
print(f"Error from Theory:     +{loss_naive - theory_loss:.4f} higher than random guessing!")

# Condition B: The Humble Sanity Check (σ=0.01)
np.random.seed(42)
W_head_scaled = np.random.randn(n_embed, v_size) * 0.01  # std = 0.01

# 1. Logits & Softmax
logits_scaled = x_final @ W_head_scaled
exp_scaled = np.exp(logits_scaled - np.max(logits_scaled, axis=-1, keepdims=True))
prob_scaled = exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)

# 2. Correct token probabilities & Loss
correct_p_scaled = prob_scaled[range(len(y)), y]
loss_scaled = -np.mean(np.log(correct_p_scaled))

print("-" * 60)
print("--- CONDITION B: HUMBLE SCALED W_head (std = 0.01) ---")
print(f"Logit Range:           [{np.min(logits_scaled):.4f}, {np.max(logits_scaled):.4f}]")
print(f"Logit Dynamic Spread:  {np.ptp(logits_scaled):.4f}")
print(f"P(correct tokens):     {np.round(correct_p_scaled, 4)}")
print(f"Mean P(correct):       {np.mean(correct_p_scaled):.4f} (Benchmark: 1/11 = {1/v_size:.4f})")
print(f"Initial Loss:          {loss_scaled:.4f}")
print(f"Error from Theory:     {abs(loss_scaled - theory_loss):.4f} (Near ZERO!)")