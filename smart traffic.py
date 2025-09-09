import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs

import numpy as np
import pandas as pd
import folium
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ------------------------
# Step 1: Generate Synthetic Dataset
# ------------------------
np.random.seed(42)

data = {
    "traffic_density": np.random.randint(10, 200, 500),   # vehicles/min
    "temperature": np.random.randint(15, 40, 500),        # °C
    "humidity": np.random.randint(20, 90, 500),           # %
    "AQI": np.random.randint(50, 300, 500)                # AQI (target)
}

df = pd.DataFrame(data)

# ------------------------
# Step 2: Preprocess
# ------------------------
X = df[["traffic_density", "temperature", "humidity"]]
y = df["AQI"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------
# Step 3: Build ANN
# ------------------------
model = Sequential([
    Dense(16, input_dim=3, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="linear")
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# ------------------------
# Step 4: Train
# ------------------------
history = model.fit(X_train, y_train, epochs=40, batch_size=16, validation_split=0.2, verbose=0)

# ------------------------
# Step 5: Evaluate
# ------------------------
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Mean Absolute Error: {mae:.2f}")

# ------------------------
# Step 6: Predictions
y_pred = model.predict(X_test).flatten()  # Flatten for plotting

plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Traffic Pollution Control: Actual vs Predicted AQI")
plt.show()

# ------------------------
# Step 7: Map Visualization with Folium
# ------------------------

# Synthetic hotspot coordinates (e.g., around New Delhi)
locations = [
    (28.6139, 77.2090),  # Delhi center
    (28.7041, 77.1025),  # North Delhi
    (28.5355, 77.3910),  # Noida
    (28.4595, 77.0266),  # Gurgaon
    (28.4089, 77.3178)   # Faridabad
]

np.random.seed(42)  # For reproducibility of hotspot AQI

# Randomly assign AQI predictions to these locations
hotspot_data = []
for loc in locations:
    traffic = np.random.randint(50, 200)
    temp = np.random.randint(20, 38)
    hum = np.random.randint(25, 80)
    
    scaled = scaler.transform([[traffic, temp, hum]])
    pred_aqi = model.predict(scaled)[0][0]
    
    hotspot_data.append((loc[0], loc[1], pred_aqi))

# Create Folium Map
m = folium.Map(location=[28.6139, 77.2090], zoom_start=10)

# Add markers
for lat, lon, aqi in hotspot_data:
    if aqi < 100:
        color = "green"
    elif aqi < 200:
        color = "orange"
    else:
        color = "red"
    
    folium.CircleMarker(
        location=[lat, lon],
        radius=10,
        popup=f"AQI: {int(aqi)}",
        color=color,
        fill=True,
        fill_color=color
    ).add_to(m)

# Save map as HTML
m.save("traffic_pollution_map.html")
print("✅ Map saved as traffic_pollution_map.html")