import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("ml13.csv")

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

model1 = LinearRegression()
model1.fit(x,y)

plt.plot(x,model1.predict(x),color = 'r')
plt.scatter(x,y,color = 'g')
plt.show()

from sklearn.preprocessing import  PolynomialFeatures
model2 = PolynomialFeatures(degree = 3)
xpoly = model2.fit_transform(x)
#print(xpoly[:,-1])

model1.fit(xpoly,y)


plt.plot(x,model1.predict(xpoly),color = 'r')
plt.scatter(x,y,color = 'g')
plt.show()
