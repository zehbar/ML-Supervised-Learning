import numpy as np
import pandas as pd

df = pd.read_csv('digit.csv')
#print(df)

x = df.iloc[:,1:]
y = df.iloc[:,0]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size = 0.25 , random_state = 0)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 280, min_samples_split = 3, min_samples_leaf = 1, max_features = 'sqrt' , criterion = 'gini' , max_depth = 150,random_state = 0)
model.fit(x_train,y_train)

pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred)*100)