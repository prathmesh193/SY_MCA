import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression

iris = sns.load_dataset("iris")
print(iris)

iris = iris[['petal_length','petal_width']]
print(iris)
print()

X=iris['petal_length']
y=iris['petal_width']

plt.scatter(X,y)
plt.xlabel("petal_length")
plt.ylabel("petal_width")
plt.title("Iris Dataset")
plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size= 0.4,random_state=23)

X_train = np.array(X_train).reshape(-1,1)
print("X_train")
print("------------------------------------------------------------------")
print(X_train)
print()
X_test = np.array(X_test).reshape(-1,1)
print("X_test")
print("------------------------------------------------------------------")
print(X_test)
print()

lr = LinearRegression()
lr.fit(X_train,y_train)
c = lr.intercept_
print("Intercept = ",c)
m=lr.coef_
print("Coefficient = ",m)
print()

y_pred_train = m*X_train + c
y_pred_train.flatten()
print("Line Equation")
print("------------------------------------------------------------------")
print(y_pred_train)
print()

y_pred_train1 = lr.predict(X_train)
print("Predicted values")
print("------------------------------------------------------------------")
print(y_pred_train1)
print()

#plt.scatter(X_train,y_train)
#plt.plot(X_train,y_pred_train1, color='red')
#plt.xlabel("petal_length" )
#plt.ylabel("petal_width")
#plt.show()

y_pred_test1 = lr.predict(X_test)
print("Predicted dataset")
print("------------------------------------------------------------------")
print(y_pred_test1)
print()
plt.scatter(X_train,y_train,color="cyan")
plt.plot(X_train,y_pred_train1, color='red')
plt.xlabel("petal_length")
plt.ylabel("petal_width")
plt.title("Training Graph")
plt.show()

plt.scatter(X_test,y_test,color='orange')
plt.plot(X_test,y_pred_test1, color='violet')
plt.xlabel("petal_length")
plt.ylabel("petal_width")
plt.title("Testing Graph")
plt.show()
