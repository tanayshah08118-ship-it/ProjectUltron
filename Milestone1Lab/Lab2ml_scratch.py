import numpy as np

# The "Bomb Test" (Experiencing Variance Explosion)

# The Unscaled Explosion (Naive Initialization)
X=np.random.randn(100,64)
W=np.random.randn(64,64)
print("-"*100)
print("The Naive Initilization")
for layer in range(30):
    W=np.random.randn(64,64)
    X= X @ W
    if layer%5==0:
        print(f"layer{layer:2d} | spread{np.std(X):.4f}")

# The Initialization (The Stabilizer)
print("-"*100)
print("The Stabilizer")
X=np.random.randn(100,64)
for layer in range(30):
    W=np.random.randn(64,64) * np.sqrt(1.0 / 64)
    Z=X@W
    X=np.maximum(0,Z)
    if layer%5==0:
        print(f"layer{layer:2d} | spread{np.std(X):.4f}")