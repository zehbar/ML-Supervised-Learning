import numpy as np
import pandas as pd

df = pd.read_csv('DigitalAd_dataset.csv')

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size = 0.25 , random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(x_train,y_train)
pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix , accuracy_score
cm =  confusion_matrix(y_test,pred)

print(cm)

print(accuracy_score(y_test,pred))

