import pandas as pd
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

dataset = load_digits()
print("Dataset keys")
print("--------------------------------------------------------------------")
print(dataset.keys())
print()
dataset.data[0].reshape(8,8)
#print(dataset)


#dataset.feature_names
dataset.target[2]
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print("Printing Dataframe")
print("--------------------------------------------------------------------")
print(df.head())
print()
#df.describe()
X = df
y = dataset.target

scaler = StandardScaler()
X_scaled =scaler.fit_transform(X);
print("Standard Scaler")
print("--------------------------------------------------------------------")
print(X_scaled)
print()

X_train,X_test,y_train,y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=30)

model = LogisticRegression()
model.fit(X_train,y_train)
z=model.score(X_test, y_test)
print("Logistic Regression")
print("--------------------------------------------------------------------")
print(z)
print()

pca = PCA(0.95)
X_pca = pca.fit_transform(X)
print("PCA of 95% variance")
print("--------------------------------------------------------------------")
print(X_pca.shape)
print()
print("PCA - Varinace Ratio")
print("--------------------------------------------------------------------")
print(pca.explained_variance_ratio_)
print()
print("PCA - Total Cpmponents")
print(pca.n_components_)
print()

X_train_pca, X_test_pca,y_train,y_test = train_test_split(X_pca, y, test_size=0.2,random_state=30)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train)
print("PCA Train - Test")
print("--------------------------------------------------------------------")
print(model.score(X_test_pca,y_test))
print()

print("PCA - 5 components")
print("--------------------------------------------------------------------")
pca=PCA(n_components =5)
X_pca = pca.fit_transform(X)
print(X_pca.shape)

plt.gray()
plt.matshow(dataset.data[2].reshape(8,8))
plt.show()
