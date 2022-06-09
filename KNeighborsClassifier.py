import numpy as np
import pandas as pd

df = pd.read_csv('salary.csv')

x = df.iloc[:,:-1].values
'''y = list()
for k in df.iloc[:,-1].values:
    if k == "<=50K":
        y.append(0) 
    else: 
        y.append(1)'''

income = set(df['income'])
df['income'] = df['income'].map({"<=50K" : 0 , ">50K" : 1}).astype(int)

y = df.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size = 0.25 , random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

error = []
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
for i in range(1,40):
    model = KNeighborsClassifier(n_neighbors = i)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    error.append(np.mean(pred != y_test))

plt.figure(figsize=(10,10))
plt.plot(range(1,40),error)
plt.show()

model = KNeighborsClassifier(n_neighbors = 1+error.index(min(error)),metric = 'minkowski' , p = 2)
model.fit(x_train,y_train)
pred = model.predict(x_test)



from sklearn.metrics import confusion_matrix , accuracy_score , roc_auc_score , roc_curve
cm = confusion_matrix(y_test,pred)
print(cm)
print(accuracy_score(y_test,pred)*100)

import matplotlib.pyplot as plt

noprob = [0 for _ in range(len(y_test))]
lsprob = model.predict_proba(x_test)
#print(lsprob)
#keep prob for +ve outcome only
lsprob = lsprob[:,1]
#print(lsprob)
#cal score
noauc = roc_auc_score(y_test , noprob)
lsauc = roc_auc_score(y_test , lsprob)

print("no skill auc = %.3f"%(noauc*100))
print("lr skill auc = %.3f"%(lsauc*100))

nsfp , nstp , _ = roc_curve(y_test , noprob)
lrfp , lrtp , _ = roc_curve(y_test , lsprob)

plt.plot(nsfp , nstp, linestyle = 'dashed',label = 'No Skill')
plt.plot(lrfp , lrtp, marker = '*',label = 'KNN')

plt.legend()
plt.show()