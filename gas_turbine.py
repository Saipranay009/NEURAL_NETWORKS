# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:45:03 2022

@author: Sai pranay
"""
#------------------------------importing---------------------------------------

import pandas as pd

gt = pd.read_csv("E:\\DATA_SCIENCE_ASS\\NEURAL NETWORKS\\gas_turbines.csv")
print(gt)
list(gt)
gt.shape
gt.info()
gt.describe()
gt.isnull().sum()
gt.hist()

gt["TEY"].value_counts()


gt1 = gt.drop(["AFDP","GTEP","TAT","CO","NOX","TIT","CDP"],axis=1)
gt1


#plot
import seaborn as sns
import matplotlib.pyplot as plt

#--------------------checking correlation--------------
gt1.corr()


plt.figure(figsize=(10,10))
sns.heatmap(gt1,annot=True)

#------------------- spiltting----------------------

x = gt1.iloc[:,0:3]
x

y = gt1.iloc[:,3]
y

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# --------------------- modelfitting--------------------
model = Sequential()
model.add(Dense(5, input_dim=3,  activation='relu')) #input layer
#model.add(Dense(3, activation='relu')) #output layer
model.add(Dense(1, activation='relu')) #output layer



#---------------------------- Compile model----------------------
model.compile(loss='msle', optimizer='adam', metrics=['msle'])


# ---------------------------Fit the model-----------------------------
history = model.fit(x, y, validation_split=0.25, epochs=400, batch_size=20)

# ----------------------evaluate the model--------------------------
scores = model.evaluate(x, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))

history.history.keys()


# summarize history for mse

plt.plot(history.history['msle'])
plt.plot(history.history['val_msle'])
plt.title('model msle')
plt.ylabel('msle')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
