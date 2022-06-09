import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

dataset = load_digits()
'''
print(dataset.data)
print(dataset.target)

print(dataset.data.shape)
print(dataset.images.shape)'''

dataimagelen = len(dataset.images)
#print(dataimagelen)

'''n = 1700   #no of sample out of samples total 1797

import matplotlib.pyplot as plt
plt.gray()
plt.matshow(dataset.images[n])
plt.show()

#print(dataset.images[n])
x = dataset.images
print(x.shape)
print("-"*100)'''
x = dataset.images.reshape((dataimagelen,-1))
#print(x.shape)
y = dataset.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size = 0.25 , random_state = 0)

'''from sklearn import svm
model = svm.SVC(kernel = 'linear')
model.fit(x_train,y_train)

n = -55

result = model.predict(dataset.images[n].reshape((1,-1)))
plt.imshow(dataset.images[n] , cmap = plt.cm.gray_r , interpolation  = 'nearest')
print(result)
print()
plt.axis('off')
plt.title('%i'%result)
plt.show() 

pred = model.predict(x_test)
print(np.concatenate((pred.reshape(len(pred),1),y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix , accuracy_score
cm =  confusion_matrix(y_test,pred)
print(cm)

print(accuracy_score(y_test,pred))'''

from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn import svm
model1 = svm.SVC(kernel='linear')
model2 = svm.SVC(kernel='rbf')
model3 = svm.SVC(gamma=0.0008)
model4 = svm.SVC(gamma=0.001,C=0.7)
model5 = svm.SVC(kernel = 'poly')


model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)
model5.fit(x_train,y_train)


y_predModel1 = model1.predict(x_test)
y_predModel2 = model2.predict(x_test)
y_predModel3 = model3.predict(x_test)
y_predModel4 = model4.predict(x_test)
y_predModel5 = model5.predict(x_test)


print("Accuracy of the Model 1: {0}%".format(accuracy_score(y_test, y_predModel1)*100))
print("Accuracy of the Model 2: {0}%".format(accuracy_score(y_test, y_predModel2)*100))
print("Accuracy of the Model 3: {0}%".format(accuracy_score(y_test, y_predModel3)*100))
print("Accuracy of the Model 4: {0}%".format(accuracy_score(y_test, y_predModel4)*100))
print("Accuracy of the Model 5: {0}%".format(accuracy_score(y_test, y_predModel5)*100))
