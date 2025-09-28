import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score,classification_report

data=load_iris()
X=data.data
y=data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
kernels=["linear","rbf","poly"]

models={k:SVC(kernel=k) for k in kernels}
for k,m in models.items():
    m.fit(X_train,y_train)

    y_pred=m.predict(X_test)
    print(f"Kernel:{k}")
    print("Accuracy:",accuracy_score(y_test,y_pred))
    print("Report:\n",classification_report(y_test,y_pred))
  
'''
Kernel:linear
Accuracy: 1.0
Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       1.00      1.00      1.00        13
           2       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45

Kernel:rbf
Accuracy: 1.0
Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       1.00      1.00      1.00        13
           2       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45

Kernel:poly
Accuracy: 0.9777777777777777
Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       1.00      0.92      0.96        13
           2       0.93      1.00      0.96        13

    accuracy                           0.98        45
   macro avg       0.98      0.97      0.97        45
weighted avg       0.98      0.98      0.98        45
'''