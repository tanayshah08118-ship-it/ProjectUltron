import numpy as np
import pandas as pd

# Model
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
print("Train shape",train.shape)
print("Test shape",test.shape)
print(train.head())

print("Train missing values\n",train.isnull().sum())
print("Test missing values\n",test.isnull().sum())

train["Age"]=train["Age"].fillna(train["Age"].median())
train["Embarked"]=train["Embarked"].fillna(train["Embarked"].mode()[0])
train=train.drop("Cabin",axis=1)

test["Age"]=test["Age"].fillna(test["Age"].median())
test["Fare"]=test["Fare"].fillna(test["Fare"].median())
test=test.drop("Cabin",axis=1)

print("Train missing fixed",train.isnull().sum())
print("Test missing fixed",test.isnull().sum())

train["Sex"]=train["Sex"].map({"female":0,"male":1})
train["Embarked"]=train["Embarked"].map({"S":0,"C":1,"Q":2})

test["Sex"]=test["Sex"].map({"female":0,"male":1})
test["Embarked"]=test["Embarked"].map({"S":0,"C":1,"Q":2})

print("Changes",train[["Sex","Embarked"]].head())

features=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
X_train=train[features].values
y_train=train["Survived"].values
X_test=test[features].values
print("X_train shape",X_train.shape)
print("y_train shape",y_train.shape)
print("X_test shape",X_test.shape)

mean=np.mean(X_train,axis=0)
std=np.std(X_train,axis=0)
X_train_scaled=(X_train-mean)/std
X_test_scaled=(X_test-mean)/std

print("Scaled Data:\n")
print("First row before Scaling:",X_train[0])
print("First row after Scaling:",X_train_scaled[0])

def sigmoid(z):
    z=np.clip(z,-500,500)
    return 1/(1+np.exp(-z))

def cost_computed(X,y,W,b):
    m=X.shape[0]
    z=np.dot(X,W)+b
    y_hat=sigmoid(z)
    eps=1e-15
    y_hat=np.clip(y_hat,eps,1-eps)
    cost=-(1/m)*np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))
    return cost
def grad_decent(X,y,W,b,lr,iter):
    m=X.shape[0]
    cost_history=[]
    for i in range(iter):
        z=np.dot(X,W)+b
        y_hat=sigmoid(z)
        error=y_hat-y
        dW=(1/m)*np.dot(X.T,error)
        db=(1/m)*np.sum(error)
        W-=lr*dW
        b-=lr*db
        if i%100==0:
            cost=cost_computed(X,y,W,b)
            cost_history.append(cost)
            print(f"iter{i:6d}|Cost{cost:.4f}")
    return W,b,cost_history

W = np.zeros(X_train_scaled.shape[1])
b = 0

W, b, cost_history = grad_decent(X_train_scaled, y_train, W, b, lr=0.1, iter=10000)

print("\nFinal weights:", np.round(W, 4))
print("Final bias:", round(b, 4))

z = np.dot(X_train_scaled, W) + b
y_hat = sigmoid(z)

y_pred = (y_hat >= 0.5).astype(int)
accuracy = np.mean(y_pred == y_train)
print(f"\nTraining Accuracy: {accuracy * 100:.2f}%")


# Testing 
z_test=np.dot(X_test_scaled,W)+b
y_test_hat=sigmoid(z_test)
test_pred=(y_test_hat>=0.5).astype(int)

submission=pd.DataFrame({
    "PassengerId":test["PassengerId"],
    "Survived":test_pred
})
submission.to_csv("submission.csv",index=False)
print("\nSubmission file created!")
print("Shape:", submission.shape)
print(submission.head(10))