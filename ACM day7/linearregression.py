import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(42)
x=2*np.random.rand(100,1)
y=4+3*x+np.random.randn(100,1)
x1=np.c_[np.ones((x.shape[0],1)),x]
b=np.linalg.inv(x1.T@x1)@x1.T@y
i,cf=b[0][0],b[1][0]
print("NumPy Implementation:")
print(f"Intercept:{i:.4f},Coefficient:{cf:.4f}")
yh=x1@b
ssr=np.sum((y-yh)**2)
sst=np.sum((y-np.mean(y))**2)
r=1-(ssr/sst)
print(f"R-squared(NumPy):{r:.4f}")
m=LinearRegression()
m.fit(x,y)
print("\nScikit-learn Implementation:")
print(f"Intercept:{m.intercept_[0]:.4f},Coefficient:{m.coef_[0][0]:.4f}")
yh2=m.predict(x)
r2=r2_score(y,yh2)
print(f"R-squared(sklearn):{r2:.4f}")
