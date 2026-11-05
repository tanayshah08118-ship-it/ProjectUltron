import numpy as np

# Stage 1: The Tokenizer (How Computers Read Words)

# Step 1: Tansforming the string into index, sorting it and finding the size of vocabulary
text="hello tokenizer"
letters=set("hello tokenizer")
char=sorted(list(letters))
v_size=len(char)
print("-"*100)
print("Charcters in text:")
print(char)
print("-"*100)
print("Vocabulary Size",v_size)
# Step 2: The Two-Way Translation Dictionaries (stoi(string to integer) and itos(integer to string)), but only for single letter
stoi ={ch:i for i , ch in enumerate(char)}
itos ={i:ch for i,ch in enumerate(char)}
print("-" * 100)
print("Testing single-letter translation:")
letter ='h'
num = stoi[letter]
print("-"*100)
print(f"The number for '{letter}' is:", num)
print(f"Translating {num} back to text:", itos[num])
# Step 3: Encoding & Decoding Entire Sentences
def encode(s):
    return [stoi[c] for c in s]
def decode(l):
    return ''.join([itos[i] for i in l])
print("-" * 100)
print("Testing Full String Tokenization:")
test_string = "hello"
encoded = encode(test_string)
decoded = decode(encoded)
print(f"Original Text: '{test_string}'")
print(f"Encoded Token IDs: {encoded}")
print(f"Decoded Back: '{decoded}'")

# Stage 2: The Embedding Layer (From IDs to Meaning)
n_embed=16
np.random.seed(42)
W_ta=np.random.randn(v_size,n_embed)
token_embeddings = W_ta[encoded]
print("-" * 100)
print("Token Embeddings Shape:", token_embeddings.shape)
print("Token Embeddings Matrix:\n", token_embeddings)
# Positional Embeddings
block_size=8
W_pos=np.random.randn(block_size,n_embed)
T=len(encoded)
pos_embeddings=W_pos[:T]
x=token_embeddings+pos_embeddings
print("-"*100)
print("Shape of x",x.shape)
print("X data: \n",x)

# Stage 3: Modern Normalization — RMSNorm (Root Mean Square Normalization)
gamma=np.ones(n_embed)
def rms_norm(x,gamma,eps=1e-5):
    rms=np.sqrt(np.mean(x**2,axis=-1,keepdims=True)+eps)
    return (x/rms)*gamma
x_norm = rms_norm(x, gamma)
print("-" * 100)
print("RMSNorm Output Shape:", x_norm.shape)
print("RMSNorm Output Sample:\n", x_norm)

# Stage 4: The Heart of the Transformer — Self-Attention

# Step 1: Linear Projections
W_q = np.random.randn(n_embed, n_embed)
W_k = np.random.randn(n_embed, n_embed)
W_v = np.random.randn(n_embed, n_embed)
Q=x_norm@W_q
K=x_norm@W_k
V=x_norm@W_v
print("-"*100)
print("Shape of Q,K and V \n",Q.shape,"\n",K.shape,"\n",V.shape)

# Step 2: Masking the Future
d_k = 16
scores = (Q @ K.T) / np.sqrt(d_k)
mask =np.tril(np.ones((T,T)))
scores[mask==0]=-np.inf
print("-"*100)
print("Scores:\n",scores)

# Step 3: Converting Raw Scores to Probabilities (Softmax)
exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
attn_out = attn_weights @ V
print("-" * 100)
print("Attention Weights (Probabilities):\n", np.round(attn_weights, 3))
print("-" * 100)
print("Row Sums (Should all be 1.0):", np.sum(attn_weights, axis=-1))
print("-" * 100)
print("Attention Output Shape:", attn_out.shape)

# Residual connection:
x+=attn_out
print("-" * 100)
print("Shape after Residual 1:", x.shape)

# Stage 5: The Feed-Forward Brain — SwiGLU
# Step 1: The Swish Activation Function
gamma_2=np.ones(n_embed)
x_norms2=rms_norm(x,gamma_2)
def swish(z):
    return z/(1.0+np.exp(-z))
d_ffn=64
W_gate=np.random.randn(n_embed,d_ffn)
W_up=np.random.randn(n_embed,d_ffn)
W_down=np.random.randn(d_ffn,n_embed)
gate=swish(x_norms2@W_gate)
up=x_norms2@W_up
hidden=gate*up
ffn_out=hidden@W_down
x+=ffn_out
print("-"*100)
print("shape of ffn_out\n",ffn_out.shape)
print("-"*100)
print("shape of x \n",x.shape)

# Stage 6: The Language Model Head (How the Model Speaks)
# Step 1: Predicting the Next Character
gamma_final=np.ones(n_embed)
x_final=rms_norm(x,gamma_final) # <-- if changed n_embed-->gamma_final the prediction changes and the model gives a reallistic prediction
W_head=np.random.randn(n_embed,v_size)
logits=x_final@W_head
print("-"*100)
print("Logits Shape (Tokens x Vocab):", logits.shape)
exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
next_token_probs = probs[-1]
predicted_id = np.argmax(next_token_probs)
predicted_char = itos[predicted_id]
print("-" * 100)
print("Input string: 'hello'")
print(f"Untrained Model's Top Prediction: '{predicted_char}' (ID: {predicted_id})")
print("Next Token Probability Distribution:\n", np.round(next_token_probs, 3))

# Stage 7: The Autoregressive Generation Loop (How LLMs Generate Sentences)
def forward(token_ids):
    token_ids=token_ids[-block_size:]
    T=len(token_ids)
    x=W_ta[token_ids]+W_pos[:T]
    x_norm1=rms_norm(x,gamma)
    Q=x_norm1@W_q
    K=x_norm1@W_k
    V=x_norm1@W_v
    scores=(Q @ K.T)/np.sqrt(d_k)
    mask= np.tril(np.ones((T,T)))
    scores[mask==0]=-np.inf
    exp_scores=np.exp(scores-np.max(scores,axis=-1,keepdims=True))
    attn_weights=exp_scores/np.sum(exp_scores,axis=-1,keepdims=True)
    attn_out=attn_weights@V
    x+=attn_out
    x_norm2 = rms_norm(x, gamma_2)
    gate = swish(x_norm2 @ W_gate)
    up = x_norm2 @ W_up
    ffn_out = (gate * up) @ W_down
    x = x + ffn_out 
    x_final = rms_norm(x, gamma_final)
    logits = x_final @ W_head
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return probs[-1]
prompt = "hello"
generated_ids = encode(prompt)

print("-" * 100)
print(f"Initial Prompt: '{prompt}'")
print("Generating 10 characters autoregressively: ", end="", flush=True)

for _ in range(10):
  next_probs = forward(generated_ids)
  next_id = np.argmax(next_probs)  # pick the most likely character
  generated_ids.append(next_id)  # append to history
  print(itos[next_id], end="", flush=True)

print()
print("-" * 100)
print("Complete Generated Text:\n", decode(generated_ids))