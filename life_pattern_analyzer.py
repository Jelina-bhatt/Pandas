import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Step 1: Create Sample Data
# -----------------------------
data = {
    'Date': pd.date_range(start='2025-09-01', periods=30),
    'Sleep_Hours': [6,7,5,8,6,7,5,6,8,7,6,5,8,7,6,7,6,8,5,6,7,8,7,6,5,7,8,6,7,8],
    'Study_Hours': [3,4,2,5,3,4,2,3,5,4,3,2,5,4,3,4,3,5,2,3,4,5,4,3,2,4,5,3,4,5],
    'Screen_Time_Hrs': [4,5,6,3,5,4,7,6,3,4,5,7,3,4,6,5,6,3,7,6,4,3,4,5,7,4,3,6,5,3],
    'Mood_Score': [6,7,5,8,6,7,4,5,8,7,6,4,8,7,5,7,6,8,5,6,7,8,7,5,4,7,8,6,7,8]
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# -----------------------------
# Step 2: Basic Overview
# -----------------------------
print("Dataset Overview:\n", df.head(), "\n")
print("Summary Statistics:\n", df.describe(), "\n")

# -----------------------------
# Step 3: Correlation Analysis
# -----------------------------
print("Correlation Matrix:\n", df.corr(), "\n")
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Daily Habits")
plt.show()

# -----------------------------
# Step 4: Weekly Trend Analysis
# -----------------------------
df['Week'] = df.index.isocalendar().week
weekly_avg = df.groupby('Week').mean(numeric_only=True)
print("Weekly Averages:\n", weekly_avg)

weekly_avg.plot(y=['Sleep_Hours', 'Study_Hours', 'Screen_Time_Hrs'], kind='line', marker='o')
plt.title("Weekly Average Routine Patterns")
plt.ylabel("Hours")
plt.show()

# -----------------------------
# Step 5: Rolling (Moving) Average for Mood
# -----------------------------
df['Mood_7Day_Avg'] = df['Mood_Score'].rolling(7).mean()

plt.figure(figsize=(10,5))
plt.plot(df.index, df['Mood_Score'], label='Daily Mood')
plt.plot(df.index, df['Mood_7Day_Avg'], label='7-Day Avg Mood', color='orange')
plt.title("Mood Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Mood Score")
plt.legend()
plt.show()

# -----------------------------
# Step 6: Insight Generation
# -----------------------------
correlations = df.corr()['Mood_Score'].sort_values(ascending=False)
print("\n✨ Mood Correlation Insights:")
for col, value in correlations.items():
    if col != 'Mood_Score':
        if value > 0.3:
            print(f"→ {col} has a positive impact on mood ({value:.2f})")
        elif value < -0.3:
            print(f"→ {col} has a negative impact on mood ({value:.2f})")

# -----------------------------
# Step 7: Save Report
# -----------------------------
df.to_csv("life_pattern_analysis.csv")
print("\n✅ Report saved as life_pattern_analysis.csv")
