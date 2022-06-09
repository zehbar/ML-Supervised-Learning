import numpy as np
import pandas as pd

df = pd.read_csv("titanicsurvival.csv")

sex = set(df['Sex'])
df['Sex'] = df['Sex'].map({'male' : 1, 'female' : 0}).astype(int)
#print(df)



x = df.drop("Survived",axis = 1)
y = df['Survived']

#print(y)

#for nan values
print(x.columns[x.isna().any()])
#will give Age
x.Age = x.Age.fillna(x.Age.mean())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size = 0.25 , random_state = 0)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train,y_train)

pred = model.predict(x_test)


from sklearn.metrics import confusion_matrix , accuracy_score
cm =  confusion_matrix(y_test,pred)

print(cm)

print(accuracy_score(y_test,pred))

