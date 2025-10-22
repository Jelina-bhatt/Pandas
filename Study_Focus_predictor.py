import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# Step 1: Create Dataset
# -----------------------------------
data = {
    'Date': pd.date_range(start='2025-09-01', periods=30),
    'Sleep_Hours': [6,7,5,8,6,7,6,5,8,7,6,5,7,8,6,7,6,5,8,7,6,7,5,6,8,7,6,5,7,8],
    'Study_Hours': [2,3,4,5,3,4,2,3,4,5,3,4,2,3,5,4,3,2,5,4,3,4,2,3,4,5,3,4,2,3],
    'Breaks_Taken': [5,4,6,3,5,4,6,5,3,4,5,6,3,5,4,6,5,3,4,6,5,4,6,3,5,4,6,5,3,4],
    'Screen_Time_Hrs': [6,5,7,4,6,5,7,6,4,5,7,6,5,4,6,5,7,6,4,5,6,5,7,6,4,5,7,6,4,5],
    'Focus_Score': [60,70,55,80,65,75,60,55,85,70,65,60,80,75,70,75,65,55,85,75,65,70,60,65,80,75,65,60,70,80]
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# -----------------------------------
# Step 2: Overview
# -----------------------------------
print("Dataset Head:\n", df.head(), "\n")
print("Summary Stats:\n", df.describe(), "\n")

# -----------------------------------
# Step 3: Correlation Matrix
# -----------------------------------
plt.figure(figsize=(7,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Habits and Focus")
plt.show()

# -----------------------------------
# Step 4: Rolling Average (Focus Trend)
# -----------------------------------
df['Focus_7Day_Avg'] = df['Focus_Score'].rolling(7).mean()

plt.figure(figsize=(10,5))
plt.plot(df.index, df['Focus_Score'], label='Daily Focus', color='blue')
plt.plot(df.index, df['Focus_7Day_Avg'], label='7-Day Average', color='orange', linewidth=2)
plt.title("Focus Trend Over 30 Days")
plt.xlabel("Date")
plt.ylabel("Focus Score")
plt.legend()
plt.show()

# -----------------------------------
# Step 5: Key Insight Extraction
# -----------------------------------
correlations = df.corr()['Focus_Score'].sort_values(ascending=False)
print("\n🔍 Top Factors Affecting Focus:\n", correlations, "\n")

best_factor = correlations.index[1]
impact_value = correlations.values[1]

print(f"✨ Most Impactful Habit: {best_factor} (correlation: {impact_value:.2f})")

# -----------------------------------
# Step 6: Predict Next Week's Focus (Simple Logic)
# -----------------------------------
latest_focus = df['Focus_Score'].iloc[-1]
avg_study = df['Study_Hours'].iloc[-7:].mean()
avg_sleep = df['Sleep_Hours'].iloc[-7:].mean()

predicted_focus = latest_focus + (avg_study * 0.5) + (avg_sleep * 1.2) - 5
print(f"\n📈 Predicted Focus Score for Next Week: {predicted_focus:.1f}/100")

if predicted_focus > 75:
    print("✅ Excellent consistency — keep up your study-sleep balance!")
elif predicted_focus > 60:
    print("💪 Good progress — improve slightly on sleep or reduce screen time.")
else:
    print("⚠️ Focus dropping — cut screen time and take healthy breaks.")

# -----------------------------------
# Step 7: Weekly Summary
# -----------------------------------
df['Week'] = df.index.isocalendar().week
weekly_focus = df.groupby('Week')['Focus_Score'].mean()

weekly_focus.plot(kind='bar', color='teal')
plt.title("Average Focus per Week")
plt.ylabel("Focus Score")
plt.show()

# -----------------------------------
# Step 8: Save Report
# -----------------------------------
df.to_csv("StudyFocus_Analyzer_Report.csv")
print("\n✅ Report saved as StudyFocus_Analyzer_Report.csv")
