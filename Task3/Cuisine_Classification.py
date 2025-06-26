#Step 1: Importing neccessary libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

#Step 2: Load the dataset
data = pd.read_csv("Dataset.csv")

#Step 3: Dropping missing values in the target column
data = data.dropna(subset=["Cuisines"])

#Step 4: Encoding categorical columns using LabelEncoder
le =  LabelEncoder()
categorical_cols = ['Cuisines', 'City', 'Currency', 'Has Online delivery', 'Switch to order menu', 'Has Table booking']

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

data['Price range'] = data['Price range'].astype(int)

#Step 5: Defining features and targets as X and y respectivley
X = data[['City', 'Switch to order menu', 'Votes', 'Has Online delivery', 'Has Table booking', 'Price range']]
y = data['Cuisines']

#Step 6: Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2) 

#Step 7: Train the model using RandomForestClassifier (using memory efficient settings)
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)

#Step 8: Predict and evaluate using test sets
y_pred = model.predict(X_test)

#Step 9: Calculating accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy", round(accuracy * 100, 2), "%")

#Classification Report
print("\nClassification Report: \n")
print(classification_report(y_test, y_pred))