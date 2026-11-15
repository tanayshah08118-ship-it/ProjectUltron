# Model
import numpy as np
import pandas as pd

# Load data:
dataset=pd.read_csv("iris.csv")
print(dataset.shape)
print(dataset.head())

features=["sepal_length","sepal_width","petal_length","petal_width"]

# Encode target:
dataset["species"]=dataset["species"].map({"setosa":0,"versicolor":1,"virginica":2})
X=dataset[features].values
y=dataset["species"].values

# Extract arrays:
print(X.shape)
print(y.shape)

# The Train/Test Split & The Data Leakage Law
from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# (Feature Standardization)
mean=np.mean(X_train,axis=0)
std=np.std(X_train,axis=0)
X_train_scaled=(X_train-mean)/std
X_test_scaled=(X_test-mean)/std

# One-Hot Encoding for Multi-Class Cross-Entropy
y_train_onehot=np.eye(3)[y_train]
y_test_onehot=np.eye(3)[y_test]

print(X_train_scaled.shape)
print(X_test_scaled.shape)
print(y_train.shape)
print(y_test.shape)
print(y_train_onehot[:3].shape)

# Parameters
n_feature=4
n_class=3
W=np.zeros((n_feature,n_class))
b=np.zeros((1,n_class))

# Probability
def softmax(z):
    exp_shifted=np.exp(z-np.max(z,axis=-1,keepdims=True))
    return exp_shifted/(np.sum(exp_shifted,axis=-1,keepdims=True))

# Loss 
def compute_loss(P, y_onehot):
    P_clipped = np.clip(P, 1e-15, 1.0 - 1e-15)
    return -np.mean(np.sum(y_onehot * np.log(P_clipped), axis=1))

# Traning Loop

lr=0.1
training=1000
for train in range(training):
    # Forward
    z=X_train_scaled@W+b
    P=softmax(z)
    loss=compute_loss(P,y_train_onehot)
    # Backward
    dz=(P-y_train_onehot)/len(y_train_onehot)
    dW=X_train_scaled.T@dz
    db=np.sum(dz,axis=0,keepdims=True)
    # Adjusting
    W-=lr*dW
    b-=lr*db
    
    if train%100==0 or train==training-1:
        print(f"Train{train:4d},Loss{loss:.4f}")

# Traning
z_test=X_test_scaled@W+b
P_test=softmax(z_test)
y_pred=np.argmax(P_test,axis=1)
accuracy=np.mean(y_pred==y_test)
print(f"\nTest Accuracy :{accuracy*100:.2f}%")

# Confusion Matrix (From Scratch)
cm = np.zeros((3, 3), dtype=int)
for true_label, pred_label in zip(y_test, y_pred):
    cm[true_label, pred_label] += 1

print("\nConfusion Matrix (Rows: True, Cols: Pred):")
print(cm)

# The Scikit-Learn Benchmark
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

print("\n"+"-"*100)
print("SCIKIT-LEARN BENCHMARK COMPARISON")
print("-"*100)

clf = LogisticRegression( max_iter=1000)
clf.fit(X_train_scaled, y_train)

sklearn_pred = clf.predict(X_test_scaled)
sklearn_acc = np.mean(sklearn_pred == y_test)

print(f"Scikit-Learn Test Accuracy: {sklearn_acc * 100:.2f}%")
print("\nScikit-Learn Confusion Matrix:")
print(confusion_matrix(y_test, sklearn_pred))

print("\nScikit-Learn Classification Report:")
print(classification_report(y_test, sklearn_pred, target_names=["setosa", "versicolor", "virginica"]))