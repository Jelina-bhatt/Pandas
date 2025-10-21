import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# Step 1: Create Activity Data
# -----------------------------------
data = {
    'Date': pd.date_range(start='2025-09-01', periods=30),
    'Travel_km': [5, 0, 8, 3, 2, 0, 10, 6, 5, 2, 0, 8, 7, 4, 1, 0, 5, 6, 10, 2, 0, 3, 4, 8, 6, 7, 2, 5, 0, 9],
    'Electricity_kWh': [3.5, 4.0, 4.2, 3.0, 3.3, 3.8, 4.5, 3.6, 3.4, 4.0, 4.2, 3.5, 3.1, 4.1, 4.3, 3.6, 3.9, 4.0, 3.8, 4.4, 4.1, 3.7, 4.5, 4.2, 3.4, 4.0, 3.7, 4.1, 3.8, 3.9],
    'Meat_Meals': [2, 1, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 2, 3, 1, 2, 2, 3, 1, 2, 2, 3, 2, 1, 2, 3, 1, 2],
    'Public_Transport': [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

# -----------------------------------
# Step 2: Define CO2 Emission Factors
# -----------------------------------
# (approximate kg CO₂ per unit)
factors = {
    'Travel_km': 0.12,          # per km by car
    'Electricity_kWh': 0.45,    # per kWh
    'Meat_Meals': 2.5,          # per meal
    'Public_Transport': 0.05    # per ride
}

# -----------------------------------
# Step 3: Compute Carbon Footprint
# -----------------------------------
def calculate_emission(row):
    travel = row['Travel_km'] * factors['Travel_km']
    electricity = row['Electricity_kWh'] * factors['Electricity_kWh']
    meat = row['Meat_Meals'] * factors['Meat_Meals']
    public = row['Public_Transport'] * factors['Public_Transport']
    return travel + electricity + meat + public

df['CO2_kg'] = df.apply(calculate_emission, axis=1)

# -----------------------------------
# Step 4: Weekly Summary
# -----------------------------------
df['Week'] = df['Date'].dt.isocalendar().week
weekly_summary = df.groupby('Week')['CO2_kg'].sum().reset_index()

# -----------------------------------
# Step 5: Visualizations
# -----------------------------------
plt.figure(figsize=(10,5))
sns.lineplot(x='Week', y='CO2_kg', data=weekly_summary, marker='o')
plt.title("Weekly Carbon Footprint Trend")
plt.ylabel("Total CO₂ (kg)")
plt.xlabel("Week Number")
plt.show()

plt.figure(figsize=(8,5))
sns.heatmap(df[['Travel_km','Electricity_kWh','Meat_Meals','Public_Transport','CO2_kg']].corr(), annot=True, cmap='Greens')
plt.title("Correlation of Activities with CO₂ Emission")
plt.show()

# -----------------------------------
# Step 6: Insight Generation
# -----------------------------------
total_emission = df['CO2_kg'].sum()
avg_daily = df['CO2_kg'].mean()
eco_score = max(0, 100 - (avg_daily * 5))  # simple logic for fun

print(f"\n🌿 Total 30-day CO₂ emission: {total_emission:.2f} kg")
print(f"🌤 Average daily emission: {avg_daily:.2f} kg")
print(f"💚 Your Eco Score: {eco_score:.0f}/100")

if df['Travel_km'].mean() > 5:
    print("🚗 Tip: Try carpooling or using public transport 2 days/week to reduce travel emissions.")
if df['Meat_Meals'].mean() > 2:
    print("🍗 Tip: Replacing one meat meal with a vegetarian option can cut emissions significantly.")
if df['Electricity_kWh'].mean() > 4:
    print("💡 Tip: Turn off unused appliances or switch to LED lighting.")
if df['Public_Transport'].mean() >= 1:
    print("🚉 Good job! Using public transport helps lower your footprint.")

# -----------------------------------
# Step 7: Save Report
# -----------------------------------
df.to_csv("EcoTracker_Report.csv", index=False)
print("\n✅ Report saved as EcoTracker_Report.csv")
