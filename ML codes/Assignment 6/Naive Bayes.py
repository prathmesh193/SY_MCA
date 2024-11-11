import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("titanic.csv")
print("Titanic Dataset")
print(df.head())
print("-------------------------------------------------------------")


df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis= 'columns', inplace=True)
print(df.head())
print("-------------------------------------------------------------")

target = df.Survived
inputs = df.drop('Survived', axis = 'columns')
dummies = pd.get_dummies(inputs.Sex)
print(dummies.head(3))
print("-------------------------------------------------------------")

inputs = pd.concat([inputs,dummies], axis='columns')
print(inputs.head(3))
print("-------------------------------------------------------------")

inputs.drop('Sex', axis='columns', inplace = True)
print(inputs.head(3))
print("-------------------------------------------------------------")

print(inputs.columns[inputs.isna().any()])
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
print(inputs.head(6))
print("-------------------------------------------------------------")

X_train, X_test, y_train, y_test = train_test_split(inputs,target, test_size = 0.2)
print("Length of X_train : ",len(X_train))
print("Length of X_test : ",len(X_test))
print("Length of inputs : ",len(inputs))
print("-------------------------------------------------------------")



model = GaussianNB()
model.fit(X_train, y_train)
print(model.score(X_test,y_test))
print("-------------------------------------------------------------")
print("X_test values : ",X_test[:10])
print()
print("y_test values : ",y_test[:10])
print("-------------------------------------------------------------")
print("Predicted Values :",model.predict(X_test[:10]))
print("Predicted Probability : ")
print(model.predict_proba(X_test[:10]))
