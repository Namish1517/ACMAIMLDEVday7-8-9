import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

data=load_iris()
X=data.data
y=data.target
feature_names=data.feature_names

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

dt=DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)

rf=RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)

y_pred_dt=dt.predict(X_test)
y_pred_rf=rf.predict(X_test)

print("Decision Tree Accuracy:",accuracy_score(y_test,y_pred_dt))
print("Random Forest Accuracy:",accuracy_score(y_test,y_pred_rf))

print("\nDecision Tree Report:\n",classification_report(y_test,y_pred_dt))
print("Random Forest Report:\n",classification_report(y_test,y_pred_rf))

importances_dt=dt.feature_importances_
importances_rf=rf.feature_importances_

x=np.arange(len(feature_names))
plt.bar(x-0.2,importances_dt,0.4,label="Decision Tree")
plt.bar(x+0.2,importances_rf,0.4,label="Random Forest")
plt.xticks(x,feature_names,rotation=45)
plt.ylabel("Importance")
plt.title("Feature Importances")
plt.legend()
plt.tight_layout()
plt.show()
'''

Decision Tree Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       1.00      1.00      1.00        13
           2       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45

Random Forest Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       1.00      1.00      1.00        13
           2       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45
'''