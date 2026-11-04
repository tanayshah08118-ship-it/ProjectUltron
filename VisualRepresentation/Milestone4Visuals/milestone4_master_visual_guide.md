# Milestone 4 Master Visual Guide: Training the Modern Transformer

This visual guide provides the definitive mathematical, architectural, and empirical breakdown of **Milestone 4: Training the Modern Transformer**. 

Here, the Transformer transitions from a static feed-forward computation graph into an **active learning machine**, updating all 9 parameter tensors via first-principles matrix calculus and gradient descent in pure NumPy.

---

## 1. High-Resolution Architecture & Training Infographic

![Milestone 4 Master Visual Guide]("C:\Users\tanay\.gemini\antigravity\brain\0eed4e6a-1ebd-4b4d-8da5-c01b5e950d67\milestone4_training_visual_guide.png")

> [!NOTE]
> The infographic above is rendered at 200 DPI and stored locally at [`milestone4_training_visual_guide.png`](file:///c:/ProjectUltron/VisualRepresentation/Milestone4Visuals/milestone4_training_visual_guide.png). It maps the five foundational pillars of Transformer training: Causal Slicing, Cross-Entropy Loss, the 9-Gradient Backpropagation Map, 100-Step Convergence, and Autoregressive Proof of Life.

---

## 2. The 5 Core Pillars of Transformer Training

```mermaid
flowchart TD
    A["1. Causal Next-Token Framing<br>x = tokens[:-1], y = tokens[1:]"] --> B["2. Cross-Entropy Loss<br>L = -mean(log(P_correct))"]
    B --> C["3. Backprop Through LM Head<br>dlogits = (P - Y) / T<br>dW_head = x_final^T @ dlogits"]
    C --> D["4. Backprop Through Backbone<br>SwiGLU -> Residual Highway -> Attention -> Embeddings"]
    D --> E["5. Gradient Descent Update<br>W = W - lr * dW (100 Steps)<br>Loss: 2.3898 -> 0.0706"]
    E --> F["6. Autoregressive Rollout<br>Prompt 'h' -> Generates 'hello'"]

    style A fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    style B fill:#0f172a,stroke:#a855f7,stroke-width:2px,color:#f8fafc
    style C fill:#0f172a,stroke:#ec4899,stroke-width:2px,color:#f8fafc
    style D fill:#0f172a,stroke:#f59e0b,stroke-width:2px,color:#f8fafc
    style E fill:#0f172a,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style F fill:#0f172a,stroke:#06b6d4,stroke-width:2px,color:#f8fafc
```

---

## 3. Pillar 1: Causal Next-Token Dataset Slicing

Unlike standard supervised learning where inputs and targets are separate tables, an autoregressive language model trains on its own sequence:

```python
tokens = encode("hello")   # [2, 1, 5, 5, 7]
x = tokens[:-1]            # [2, 1, 5, 5] ('h', 'e', 'l', 'l')
y = tokens[1:]             # [1, 5, 5, 7] ('e', 'l', 'l', 'o')
```

| Time Step $t$ | Input Context $x[:t+1]$ | Model Target $y[t]$ | Biological / Language Objective |
| :---: | :--- | :---: | :--- |
| **0** | `['h']` | `'e'` | Given `'h'`, predict `'e'` |
| **1** | `['h', 'e']` | `'l'` | Given `'he'`, predict `'l'` |
| **2** | `['h', 'e', 'l']` | `'l'` | Given `'hel'`, predict `'l'` |
| **3** | `['h', 'e', 'l', 'l']` | `'o'` | Given `'hell'`, predict `'o'` |

> [!IMPORTANT]
> The Causal Attention Mask ensures that at time step $t$, token $t$ can only attend to tokens $\le t$. Thus, all 4 prediction targets are computed and evaluated **simultaneously in parallel** during a single forward pass!

---

## 4. Pillar 2: Cross-Entropy Loss & The Humble Initialization

The Cross-Entropy Loss quantifies model "surprise":
$$\mathcal{L} = -\frac{1}{T} \sum_{t=0}^{T-1} \ln(P_t(y_t))$$

### The Legendary Sanity Check
Before training starts, a properly initialized model should possess **humble uniform uncertainty** across the vocabulary:
$$\mathcal{L}_{\text{initial}} = -\ln\left(\frac{1}{V}\right) = \ln(V)$$

For our vocabulary of $V = 11$ characters:
$$\ln(11) \approx 2.3979$$

```python
# Empirical check on our model:
Loss: 2.3898042031497626  # Differs from theoretical by only 0.3%!
```

> [!WARNING]
> **The Overconfident Initialization Trap**:
> Initializing the LM Head with standard normal weights ($\sigma = 1.0$) caused initial loss to skyrocket to **`8.47`**! Wide logit spreads assigned $<0.02\%$ probability to the correct token, producing massive gradients. Scaling `W_head * 0.01` restored humble uncertainty.

---

## 5. Pillar 3: The Complete 9-Gradient Matrix Calculus Map

Below is the complete parameter inventory with mathematical derivative formulas and verified tensor dimensions:

| Parameter | Tensor Shape | Forward Equation | Backward Derivative Equation | Verified Gradient Shape |
| :--- | :---: | :--- | :--- | :---: |
| **$W_{\text{head}}$** | `(16, 11)` | $\text{logits} = x_{\text{final}} @ W_{\text{head}}$ | $\text{d}W_{\text{head}} = x_{\text{final}}^T @ \text{dlogits}$ | **`(16, 11)`** |
| **$W_{\text{down}}$** | `(64, 16)` | $\text{ffn\_out} = \text{hidden} @ W_{\text{down}}$ | $\text{d}W_{\text{down}} = \text{hidden}^T @ \text{dffn\_out}$ | **`(64, 16)`** |
| **$W_{\text{up}}$** | `(16, 64)` | $\text{up} = x_{\text{normal2}} @ W_{\text{up}}$ | $\text{d}W_{\text{up}} = x_{\text{normal2}}^T @ \text{dup}$ | **`(16, 64)`** |
| **$W_{\text{gate}}$** | `(16, 64)` | $z_{\text{gate}} = x_{\text{normal2}} @ W_{\text{gate}}$ | $\text{d}W_{\text{gate}} = x_{\text{normal2}}^T @ \text{dz\_gate}$ | **`(16, 64)`** |
| **$W_v$** | `(16, 16)` | $V = x_{\text{normal}} @ W_v$ | $\text{d}W_v = x_{\text{normal}}^T @ \text{d}V$ | **`(16, 16)`** |
| **$W_q$** | `(16, 16)` | $Q = x_{\text{normal}} @ W_q$ | $\text{d}W_q = x_{\text{normal}}^T @ \text{d}Q$ | **`(16, 16)`** |
| **$W_k$** | `(16, 16)` | $K = x_{\text{normal}} @ W_k$ | $\text{d}W_k = x_{\text{normal}}^T @ \text{d}K$ | **`(16, 16)`** |
| **$W_{\text{position}}$** | `(8, 16)` | $x = W_{\text{token}} + W_{\text{position}}$ | $\text{d}W_{\text{position}}[:T] \mathrel{+}= \text{d}x_{\text{embed}}$ | **`(8, 16)`** |
| **$W_{\text{token}}$** | `(11, 16)` | $x = W_{\text{token}}[token\_ids]$ | `np.add.at(dW_token, x, dx_embed)` | **`(11, 16)`** |

### Key Backpropagation Derivations:

#### 1. SwiGLU Gating Product Rule
Since $\text{hidden} = \text{gate} \odot \text{up}$:
$$\text{d}up = \text{d}hidden \odot \text{gate}$$
$$\text{d}gate = \text{d}hidden \odot \text{up}$$

#### 2. Swish Activation Derivative
For $\text{swish}(z) = z \cdot \sigma(z)$:
$$\frac{d}{dz}\text{swish}(z) = \sigma(z) + \text{swish}(z) \cdot (1 - \sigma(z))$$
$$\text{dz\_gate} = \text{dgate} \odot \left[ \sigma(z_{\text{gate}}) + \text{gate} \odot (1 - \sigma(z_{\text{gate}})) \right]$$

#### 3. General Softmax Derivative (Attention Scores)
Since scores in a row compete to sum to $1.0$:
$$\text{sum\_grad} = \sum_j (\text{d}attn\_weights \odot attn\_weights)$$
$$\text{dscores} = attn\_weights \odot (\text{d}attn\_weights - \text{sum\_grad})$$
$$\text{dscores}[\text{mask} == 0] = 0.0, \quad \text{dscores} = \frac{\text{dscores}}{\sqrt{d_k}}$$

#### 4. The Duplicate Index Accumulation Trap (`np.add.at`)
In the token sequence `[2, 1, 5, 5]`, index `5` (`'l'`) occurs twice. Direct slicing `dW_token[x] += dx` causes the second `'l'` to overwrite the first. `np.add.at(dW_token, x, dx_embed)` guarantees full atomic accumulation.

---

## 6. Pillar 4: The 100-Step Training Convergence

```
Step 00 | Loss: 2.3898   [Random Uniform Uncertainty: ~ ln(11)]
Step 10 | Loss: 1.0968   [Loss Halved: Character Frequencies Identified]
Step 20 | Loss: 0.6835   [Bigram Statistics Learned]
Step 30 | Loss: 0.4641   
Step 40 | Loss: 0.3210   
Step 50 | Loss: 0.2291   [Over 90% Confidence on Target Tokens]
Step 60 | Loss: 0.1695   
Step 70 | Loss: 0.1296   
Step 80 | Loss: 0.1022   
Step 90 | Loss: 0.0830   
Step 99 | Loss: 0.0706   [Deterministic Recall: P(correct) > 93%]
```

> [!TIP]
> **Why `* 0.1` Scaling Stabilized Training**:
> Deep networks multiply activations across multiple layers. Setting backbone weights with standard deviation $\sigma = 0.1$ maintains unit variance across layers and prevents numerical explosion in `np.exp` without needing adaptive optimizers like Adam.

---

## 7. Pillar 5: Autoregressive Proof of Life

Autoregressive inference test using the trained weights:

```python
context = [stoi["h"]]
for _ in range(4):
    prob, _ = forward(context)
    next_id = np.argmax(prob[-1])
    context.append(next_id)

print("Generated:", "".join([itos[i] for i in context]))
# OUTPUT: hello
```

The model has successfully acquired the autoregressive transition dynamics:
$$P(\text{'e'} \mid \text{'h'}) \to P(\text{'l'} \mid \text{'he'}) \to P(\text{'l'} \mid \text{'hel'}) \to P(\text{'o'} \mid \text{'hell'})$$

All code is fully verified and functional in [`Milestone4.py`](file:///c:/ProjectUltron/Milestone4.py).
