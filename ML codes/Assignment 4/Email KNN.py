import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("mail_data.csv")
print(df)
print("------------------------------------------------------------------------")

# Handle missing values
data = df.where(pd.notnull(df), '')
print(data.head(20))
print("------------------------------------------------------------------------")

# Display data info
data.info()
data.shape

# Encode categories
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

X = data['Message']
Y = data['Category']

print(X)
print()
print(Y)
print("------------------------------------------------------------------------")

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print("X shape:", X.shape)
print("X_train_shape:", X_train.shape)
print("X_test_shape:", X_test.shape)
print("------------------------------------------------------------------------")

print("Y shape:", Y.shape)
print("Y_train_shape:", Y_train.shape)
print("Y_test_shape:", Y_test.shape)
print("------------------------------------------------------------------------")

# Feature extraction using TF-IDF
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print(X_train_features)
print("------------------------------------------------------------------------")

# Create and train the model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Predictions and accuracy on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data:', accuracy_on_training_data)
print("------------------------------------------------------------------------")
print()

# Predictions and accuracy on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print("Accuracy on testing data:", accuracy_on_test_data)
print("------------------------------------------------------------------------")

# Input for prediction
#input_your_mail = ["Subject: Project Update Hi Team, I hope this message finds you well. I wanted to give you an update on the current status of the project. We are on track to meet our deadlines, and I appreciate everyone's hard work. Let’s discuss our next steps in the meeting scheduled for Thursday.Best,Emily "]
input_your_mail = ["SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"]

input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)
print(prediction)
print("------------------------------------------------------------------------")

# Output prediction result
if prediction[0] == 1:
    print("Ham Mail")
else:
    print("Spam Mail")
