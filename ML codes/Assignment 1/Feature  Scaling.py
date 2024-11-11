import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
 
#importing the dataset
dataset = pd.read_csv('data.csv')
print(dataset)
print("-------------------------------------------------------------------")

#handling missing data
x=dataset['Income'].median()
dataset["Income"].fillna(x, inplace = True)
#y=dataset['Age'].mean()
dataset['Age'].fillna(40, inplace=True)

z= dataset['Region'].mode()
dataset['Region'].fillna(z,inplace=True)

#Handling of categorical data
label_encoder= preprocessing.LabelEncoder()
dataset['Categorical data'] = label_encoder.fit_transform(dataset['Region'])
dataset['Region'].unique()
print(dataset)
print("-------------------------------------------------------------------")

#Splitting the dataset into training and testing datasets.
X= dataset['Age'] 
Y=dataset['Income'] 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y , random_state=104,  test_size=0.20) 
  
print('X_train : ') 
print(X_train.head()) 
print('') 
print('X_test : ') 
print(X_test.head()) 
print('') 
print('Y_train : ') 
print(Y_train.head()) 
print('') 
print('Y_test : ') 
print(Y_test.head())
print("-------------------------------------------------------------------")

#Feature Scaling

scaler = MinMaxScaler()
X_scaled = dataset[['Age', 'Categorical data']]
df_scaled = scaler.fit_transform(X_scaled)
print("Scaled Dataset Using MinMaxScaler")
print(df_scaled)
