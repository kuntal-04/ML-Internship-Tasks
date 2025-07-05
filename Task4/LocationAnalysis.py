#Step 1: Importing neccessary libraries 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import folium 

# Step 2: Load the dataset
data = pd.read_csv("Dataset.csv")

# Step 3: Drop rows with missing location or rating info
data = data.dropna(subset=['Latitude', 'Longitude', 'City', 'Aggregate rating'])

# Step 4: Basic Scatter Plot - Restaurant Distribution by Coordinates
plt.figure(figsize=(10,6))
sns.scatterplot(x='Longitude', y='Latitude', data=data, alpha=0.5)
plt.title('Restaurant Distribution by Location')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

# Step 5: Top 10 Cities with Most Restaurants
plt.figure(figsize=(10,6))
top_cities = data['City'].value_counts().head(10)
sns.barplot(x=top_cities.values, y=top_cities.index, palette='coolwarm')
plt.title("Top 10 Cities by Restaurant Count")
plt.xlabel("Number of Restaurants")
plt.ylabel("City")
plt.show()

# Step 6: Average Rating per City (Top 10 by Count)
city_rating = data.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=city_rating.values, y=city_rating.index, palette='viridis')
plt.title("Top 10 Cities by Average Rating")
plt.xlabel("Average Rating")
plt.ylabel("City")
plt.show()

# Step 7: Average Price Range per City
city_price = data.groupby('City')['Price range'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=city_price.values, y=city_price.index, palette='magma')
plt.title("Top 10 Cities by Average Price Range")
plt.xlabel("Average Price Range")
plt.ylabel("City")
plt.show()

# Step 8: Create an interactive map with Folium
map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
restaurant_map = folium.Map(location=map_center, zoom_start=2)

# Add 100 random restaurants to the map
for _, row in data.sample(100).iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['Restaurant Name']} ({row['City']}) - Rating: {row['Aggregate rating']}"
    ).add_to(restaurant_map)

# Save the map to an HTML file
restaurant_map.save("restaurant_map.html")
print("🌍 Interactive map saved as restaurant_map.html")
