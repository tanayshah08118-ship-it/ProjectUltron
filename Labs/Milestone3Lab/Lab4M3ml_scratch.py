import numpy as np

def swish(z):
    return z/(1+np.exp(-z))
z_closed=swish(-5.0)
z_open=swish(5.0)
print(z_closed,z_open)

# The Repetition Trap vs Temperature Sampling
logits=np.array([1.0,3.5,0.5,2.0,-1.0])
def temp_softmax(logits,temp=1.0):
    scaled_logits=logits/temp
    exp_logits=np.exp(scaled_logits-np.max(scaled_logits))
    return exp_logits /np.sum(exp_logits)
prob_cold=temp_softmax(logits,temp=0.2)
prob_normal=temp_softmax(logits,temp=1.0)
prob_hot=temp_softmax(logits,temp=2.0)
print("-"*100)
print("Cold (T=0.2):  ", np.round(prob_cold, 3))
print("-"*100)
print("Normal (T=1.0):", np.round(prob_normal, 3))
print("-"*100)
print("Hot (T=2.0):   ", np.round(prob_hot, 3))

# The Final Payoff — Breaking the Trap with Sampling
gready_pick=[np.argmax(prob_normal)for _ in range(20)]
sampled_picks=[np.random.choice(len(logits),p=prob_normal)for _ in range(20)]
print("-"*100)
print("Gready picks\n",gready_pick)
print("Sample picks\n",sampled_picks)