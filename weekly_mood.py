import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Create sample data
data = {
    "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "Mood": [7, 6, 8, 5, 9, 8, 6],              # Mood rating (1–10)
    "Sleep_Hours": [6.5, 5, 8, 4.5, 7, 7.5, 6], # Sleep per day (hrs)
    "Productivity": [6, 5, 8, 4, 9, 7, 5],      # Productivity (1–10)
}

df = pd.DataFrame(data)

# Step 2: Display the data
print("🧾 Weekly Mood & Productivity Data:\n")
print(df)
print("\n-------------------------------------\n")

# Step 3: Basic summary
print("📘 Data Summary:\n")
print(df.describe())
print("\n-------------------------------------\n")

# Step 4: Correlation analysis (fix included)
print("🔗 Correlation Matrix:")
print(df.corr(numeric_only=True))   # ✅ Fixed line
print("\n-------------------------------------\n")

# Step 5: Finding insights
best_day = df.loc[df["Productivity"].idxmax(), "Day"]
worst_day = df.loc[df["Productivity"].idxmin(), "Day"]

avg_sleep = df["Sleep_Hours"].mean()
avg_mood = df["Mood"].mean()
avg_productivity = df["Productivity"].mean()

print(f"🔥 Most Productive Day: {best_day}")
print(f"💤 Least Productive Day: {worst_day}")
print(f"📊 Average Sleep Hours: {avg_sleep:.2f}")
print(f"😊 Average Mood: {avg_mood:.2f}")
print(f"⚙️ Average Productivity: {avg_productivity:.2f}")
print("\n-------------------------------------\n")

# Step 6: Add extra insights columns
df["Mood_vs_Productivity"] = df["Mood"] - df["Productivity"]
df["Sleep_Quality"] = df["Sleep_Hours"].apply(lambda x: "Low" if x < 6 else "Good")

print("🧮 Updated DataFrame with Calculated Columns:\n")
print(df)
print("\n-------------------------------------\n")

# Step 7: Smart daily insights
for i, row in df.iterrows():
    if row["Sleep_Hours"] < 6:
        print(f"😴 On {row['Day']}, low sleep ({row['Sleep_Hours']} hrs) may have affected productivity ({row['Productivity']}).")

# Step 8: Mood-productivity relationship
correlation = df["Mood"].corr(df["Productivity"])
if correlation > 0.6:
    print("\n💡 Insight: You’re clearly more productive when you’re in a good mood!")
elif correlation < 0:
    print("\n💡 Insight: You tend to work better even when not in a great mood.")
else:
    print("\n💡 Insight: Mood doesn’t have much effect on your productivity.")
print("\n-------------------------------------\n")

# Step 9: Visualization
plt.figure(figsize=(10,6))
plt.plot(df["Day"], df["Mood"], marker='o', label='Mood', color='orange')
plt.plot(df["Day"], df["Productivity"], marker='o', label='Productivity', color='green')
plt.plot(df["Day"], df["Sleep_Hours"], marker='o', label='Sleep Hours', color='blue')

plt.title("🧠 Weekly Mood, Sleep & Productivity Analysis", fontsize=14)
plt.xlabel("Day of Week")
plt.ylabel("Scale / Hours")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 10: Save the analysis
df.to_csv("mood_productivity_analysis.csv", index=False)
print("✅ Analysis saved as 'mood_productivity_analysis.csv'")
