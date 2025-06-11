# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
path = "C:\\Users\\DELL\\Desktop\\python codes\\ML Internship\\Task1\\Dataset.csv"
data = pd.read_csv(path)

# Drop rows with missing 'Cuisines' (Cleaning missing data)
data = data.dropna(subset=['Cuisines'])

# Droping irrelevant columns
data = data.drop(columns=['Restaurant ID', 'Restaurant Name', 'Address', 'Locality', 'Locality Verbose', 'Rating color', 'Rating text'])

# Encoding categorical features
label_encoder = LabelEncoder()
for col in ['City', 'Cuisines', 'Currency', 'Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']:
    data[col] = label_encoder.fit_transform(data[col])

# Define features and targets as X and y
X = data.drop(['Aggregate rating'], axis=1)
y = data['Aggregate rating']

# Splitting data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model using LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate 
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Plot feature importance
importance = pd.Series(model.coef_, index=X.columns)
importance.sort_values().plot(kind='barh', figsize=(10, 6), title='Feature Importance')
plt.tight_layout()
plt.show()
