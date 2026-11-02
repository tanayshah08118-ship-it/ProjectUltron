import numpy as np 

# The Positional Blindspot Trap
text_a = "hello"
text_b = "olleh"  
char =sorted(list(set(text_a)))
v_size=len(char)
stoi={ch:i for i,ch in enumerate(char)}
encode=lambda s:[stoi[c] for c in s]
n_embed=16
np.random.seed(42)
W_ta=np.random.randn(v_size,n_embed)
token_a=encode(text_a)
token_b=encode(text_b)
x_a=W_ta[token_a]
x_b=W_ta[token_b] # <-- without the postionial token the postion of h in both the text is 0.0
dist_ta = np.linalg.norm(x_a[0] - x_b[4])
print("\nExperiment 1] Distance without Positional Embeddings:")
print(f"Distance between 'h' (pos 0) and 'h' (pos 4): {dist_ta:.6f}")

# Now Add Positional Embeddings
W_pos=np.random.randn(8,n_embed)
x_a_pos=x_a+W_pos[:len(token_a)]
x_b_pos=x_b+W_pos[:len(token_b)]
dist_pos= np.linalg.norm(x_a_pos[0]-x_b_pos[4])
print("-"*100)
print("With Positional Embeddings:\n",dist_pos)

W_q = np.random.randn(n_embed, n_embed) # <--Create projection matrices
W_k = np.random.randn(n_embed, n_embed) # <--Create projection matrices
Q_raw = x_a @ W_q
K_raw = x_a @ W_k
score_l1_raw = Q_raw[0] @ K_raw[2]  # 'h' looking at 1st 'l'
score_l2_raw = Q_raw[0] @ K_raw[3]  # 'h' looking at 2nd 'l'
diff_raw = abs(score_l1_raw - score_l2_raw)
Q_pos = x_a_pos @ W_q
K_pos = x_a_pos @ W_k
score_l1_pos = Q_pos[0] @ K_pos[2]
score_l2_pos = Q_pos[0] @ K_pos[3]
diff_pos = abs(score_l1_pos - score_l2_pos)
print("-" * 100)
print("ATTENTION SYMMETRY TEST (Token 0 'h' attending to 1st 'l' vs 2nd 'l'):")
print(f"Attention Score Difference WITHOUT Positions: {diff_raw:.6f}")
print(f"Attention Score Difference WITH Positions:    {diff_pos:.6f}")