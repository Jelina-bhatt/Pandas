# -----------------------------------------------
# 🌿 Forest Fire Risk Logic Predictor
# Author: Jelina Bhatt
# -----------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

# -----------------------------------------------
# 1️⃣ Data Generation or Loading
# -----------------------------------------------
if not os.path.exists('forestfires.csv'):
    data = {
        'temp': np.random.randint(10, 45, 300),
        'RH': np.random.randint(10, 90, 300),
        'wind': np.random.randint(5, 35, 300),
        'rain': np.random.uniform(0, 10, 300)
    }
    df = pd.DataFrame(data)
    df.to_csv('forestfires.csv', index=False)
    print("✅ Synthetic dataset created as 'forestfires.csv'")
else:
    df = pd.read_csv('forestfires.csv')
    print("✅ Dataset loaded successfully")

# -----------------------------------------------
# 2️⃣ Logical Fire Risk Classification
# -----------------------------------------------
def logical_risk(temp, RH, wind, rain):
    if temp > 30 and RH < 30:
        return 'High'
    elif wind > 25 and RH < 40:
        return 'High'
    elif rain > 5:
        return 'Low'
    elif temp < 15 and RH > 60:
        return 'Low'
    else:
        return 'Medium'

df['fire_risk'] = df.apply(lambda row: logical_risk(row['temp'], row['RH'], row['wind'], row['rain']), axis=1)

print("\n🔥 Sample of data with logical risk classification:")
print(df.head())

# -----------------------------------------------
# 3️⃣ Feature and Target Selection
# -----------------------------------------------
X = df[['temp', 'RH', 'wind', 'rain']]
y = df['fire_risk']

# -----------------------------------------------
# 4️⃣ Train-Test Split
# -----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# -----------------------------------------------
# 5️⃣ Random Forest Model
# -----------------------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------------------------
# 6️⃣ Evaluation
# -----------------------------------------------
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")

# -----------------------------------------------
# 7️⃣ Visualization
# -----------------------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='temp', y='RH', hue='fire_risk', palette='coolwarm', s=80)
plt.title("🔥 Fire Risk by Temperature & Humidity")
plt.xlabel("Temperature (°C)")
plt.ylabel("Relative Humidity (%)")
plt.grid(True)
plt.show()

# -----------------------------------------------
# 8️⃣ Logical Explanation Function
# -----------------------------------------------
def explain_risk(temp, RH, wind, rain):
    if temp > 30 and RH < 30:
        return "🔥 High risk: Hot & dry environment — low humidity accelerates ignition."
    elif wind > 25 and RH < 40:
        return "⚠️ High risk: Strong winds + dry air increase fire spread potential."
    elif rain > 5:
        return "💧 Low risk: Recent rainfall reduces fire chances."
    elif temp < 15 and RH > 60:
        return "🌿 Low risk: Cool & moist conditions prevent fire."
    else:
        return "🌤️ Medium risk: Moderate environmental conditions."

# -----------------------------------------------
# 9️⃣ Test Logical Explanation
# -----------------------------------------------
test_conditions = [
    {'temp': 35, 'RH': 25, 'wind': 10, 'rain': 0},
    {'temp': 20, 'RH': 75, 'wind': 15, 'rain': 2},
    {'temp': 28, 'RH': 45, 'wind': 30, 'rain': 0.5},
    {'temp': 12, 'RH': 80, 'wind': 5, 'rain': 7}
]

print("\n🧠 Logical Fire Risk Explanations:")
for cond in test_conditions:
    result = explain_risk(cond['temp'], cond['RH'], cond['wind'], cond['rain'])
    print(f"Temp: {cond['temp']}°C | RH: {cond['RH']}% | Wind: {cond['wind']} km/h | Rain: {round(cond['rain'],1)} mm -> {result}")

# -----------------------------------------------
# ✅ End of Program
# -----------------------------------------------
