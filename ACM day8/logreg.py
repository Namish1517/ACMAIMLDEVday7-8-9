import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix



np.random.seed(1)
x=np.random.randn(100,2)
y=(x[:,0]+x[:,1]>0).astype(int).reshape(-1,1)

x1=np.c_[np.ones((x.shape[0],1)),x]
def sig(z):
    return 1/(1+np.exp(-z))
def fit(x,y,lr=0.1,ep=1000):
    b=np.zeros((x.shape[1],1))
    for _ in range(ep):
        p=sig(x@b)

        g=x.T@(p-y)/len(y)
        b-=lr*g
    return b
b=fit(x1,y)
p=sig(x1@b)
yhat=(p>=0.5).astype(int)
a=accuracy_score(y,yhat)
p1=precision_score(y,yhat)
r=recall_score(y,yhat)
f=f1_score(y,yhat)
c=confusion_matrix(y,yhat)
print("Accuracy:",a)
print("Precision:",p1)
print("Recall:",r)
print("F1:",f)
print("Confusion Matrix:\n",c)
