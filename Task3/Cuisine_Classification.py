# Step 1: Importing necessary libraries 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix

# Step 2: Load the dataset
data = pd.read_csv("Dataset.csv")

# Step 3: Drop rows with missing cuisines
data = data.dropna(subset=["Cuisines"])

# Step 4: Group similar cuisines into broader categories
def group_cuisine(cuisine):
    cuisine = cuisine.lower()
    if 'indian' in cuisine:
        return 'Indian'
    elif 'chinese' in cuisine:
        return 'Chinese'
    elif 'japanese' in cuisine or 'sushi' in cuisine:
        return 'Japanese'
    elif 'american' in cuisine or 'burger' in cuisine or 'bbq' in cuisine:
        return 'American'
    elif 'italian' in cuisine or 'pizza' in cuisine:
        return 'Italian'
    elif 'thai' in cuisine:
        return 'Thai'
    elif 'mexican' in cuisine:
        return 'Mexican'
    elif 'seafood' in cuisine:
        return 'Seafood'
    else:
        return 'Other'

data['Cuisine_Group'] = data['Cuisines'].apply(group_cuisine)

# Step 5: Encode categorical columns
le = LabelEncoder()
categorical_cols = ['Cuisine_Group', 'City', 'Currency', 'Has Online delivery', 'Switch to order menu', 'Has Table booking']

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

data['Price range'] = data['Price range'].astype(int)

# Step 6: Define features (X) and target (y)
X = data[['City', 'Switch to order menu', 'Votes', 'Has Online delivery', 'Has Table booking', 'Price range']]
y = data['Cuisine_Group']

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Step 8: Train the model
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Step 9: Make predictions and evaluate
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

#Step 10: Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, cmap='Blues', xticklabels=False, yticklabels=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()