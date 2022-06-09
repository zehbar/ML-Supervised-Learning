import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

df = load_iris()
#print(df)

x = pd.DataFrame(df.data , columns = df.feature_names)
#print(x)
y = df.target

accuracy = []

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size = 0.25 , random_state = 0)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

for i in range(1,10):
    model = DecisionTreeClassifier(max_depth = i, random_state = 0)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    score = accuracy_score(y_test,pred)
    accuracy.append(score)
'''
plt.figure(figsize = (10,10))
plt.plot(range(1,10),accuracy)
plt.show()
'''

model1 = DecisionTreeClassifier(criterion = 'gini' , splitter = 'random' , max_depth = 4, random_state = 0)
#criterion can be gini
model1.fit(x_train,y_train)
pred1 = model.predict(x_test)
print(accuracy_score(y_test,pred1))
