import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report

# Load the dataset
cell_df = pd.read_csv('cell_samples.csv')
print(cell_df.head())
print("----------------------------------------------------------------------------------------------")
print(cell_df.shape)
print("----------------------------------------------------------------------------------------------")
print(cell_df.size)
print("----------------------------------------------------------------------------------------------")
print(cell_df.count())
print("----------------------------------------------------------------------------------------------")
print(cell_df['Class'].value_counts())
print("----------------------------------------------------------------------------------------------")

# Filter data for benign and malignant
benign_df = cell_df[cell_df['Class'] == 2].head(200)
malignant_df = cell_df[cell_df['Class'] == 4].head(200)

# Scatter plot
axes = benign_df.plot(kind='scatter', x='Clump', y='UnifSize', color='blue', label='Benign')
malignant_df.plot(kind='scatter', x='Clump', y='UnifSize', color='red', label='Malignant', ax=axes)
plt.show()

print(cell_df.dtypes)
print("----------------------------------------------------------------------------------------------")

# Convert 'BareNuc' to numeric and handle non-numeric values
cell_df['BareNuc'] = pd.to_numeric(cell_df['BareNuc'], errors='coerce')
print(cell_df.dtypes)
print("----------------------------------------------------------------------------------------------")

# Drop rows with NaN values if any
cell_df = cell_df.dropna()

# Define features and target
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
x = np.asarray(feature_df)
y = np.asarray(cell_df['Class'])
print("x:", x[0:5])
print("y:", y[0:5])
print("----------------------------------------------------------------------------------------------")

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
print("X_train : ", x_train.shape)
print("Y_train : ", y_train.shape)
print("X_test : ", x_test.shape)
print("Y_test : ", y_test.shape)
print("----------------------------------------------------------------------------------------------")

# Train and evaluate SVM model
classifier = svm.SVC(kernel='linear', C=2)
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)

# Print classification report
print(classification_report(y_test, y_predict))
