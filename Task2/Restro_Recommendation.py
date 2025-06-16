#importing neccessary libraries
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Step 2: Load and Preprocess the Dataset
df = pd.read_csv("Dataset.csv") 

# Drop missing values in key columns
df = df.dropna(subset=['Cuisines'])

# Encode categorical columns
le = LabelEncoder()
df['Cuisines'] = le.fit_transform(df['Cuisines'])
df['City'] = le.fit_transform(df['City'])
df['Currency'] = le.fit_transform(df['Currency'])
df['Has Online delivery'] = le.fit_transform(df['Has Online delivery'])
df['Price range'] = df['Price range'].astype(int)

# Step 3: Select Features for Recommendation Criteria
features = df[['Cuisines', 'City', 'Price range', 'Has Online delivery']]

# Step 4: Compute Similarity Matrix (Content-Based Filtering)
similarity_matrix = cosine_similarity(features)

# Step 5: Recommendation Function Based on Restaurant Index
def recommend_restaurants_by_index(index, num_recommendations=5):
    scores = list(enumerate(similarity_matrix[index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommendations = []
    for i, score in sorted_scores[1:num_recommendations+1]:  # This will skip the same restaurant
        recommendations.append((df.iloc[i]['Restaurant Name'], round(score, 2)))
    return recommendations

# Test the system with a sample restaurant index
sample_index = 10
print(f"\nUser likes: {df.iloc[sample_index]['Restaurant Name']}")
print("Top 5 similar restaurants based on content:")
for name, score in recommend_restaurants_by_index(sample_index):
    print(f"{name} | Similarity Score: {score}")

# Step 6: Recommendation Function Based on User Preferences
# User preference inputs
user_cuisine = 'Japanese'
user_city = 'Makati City'
user_price = 3
user_delivery = 'Yes'

# Encode user input to match dataset encoding
user_cuisine_encoded = le.transform([user_cuisine])[0] if user_cuisine in le.classes_ else 0
user_city_encoded = le.transform([user_city])[0] if user_city in le.classes_ else 0
user_delivery_encoded = 1 if user_delivery.lower() == 'yes' else 0

# Creating user input vector
user_input = pd.DataFrame([[user_cuisine_encoded, user_city_encoded, user_price, user_delivery_encoded]],
                          columns=['Cuisines', 'City', 'Price range', 'Has Online delivery'])

# Compute similarity of user input to all restaurants
user_similarity = cosine_similarity(user_input, features)
scores = list(enumerate(user_similarity[0]))
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

# Display top 5 recommendations
print("\nTop 5 restaurant recommendations based on user preferences:")
for i, score in sorted_scores[:5]:
    print(f"{df.iloc[i]['Restaurant Name']} | Score: {round(score, 2)}")
