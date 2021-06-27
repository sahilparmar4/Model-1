

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv("hiring.csv")
df

df['test_score'].fillna(df['test_score'].mean(),inplace=True)
df['experience'].fillna(0,inplace=True)
df

def string_to_number(word):
  dict = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
          'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,0:0}
  return dict[word]

df['experience'] = df['experience'].apply(lambda x: string_to_number(x))
df

X = df.iloc[:,:3]
Y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.1,random_state=5)

from sklearn.linear_model import LinearRegression
mymodel = LinearRegression()




mymodel.fit(x_train,y_train)

y_pred = mymodel.predict(x_test)

y = mymodel.predict([[3,5,7]])

y

y_pred

import pickle
pickle.dump(mymodel,open("model.pkl","wb"))







