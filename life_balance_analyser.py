import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

print("🌿 Welcome to the Life Balance Analyzer 🌿\n")

# Load data
try:
    df = pd.read_csv("life_balance.csv")
except FileNotFoundError:
    print("No data file found. Creating a new one.")
    df = pd.DataFrame(columns=["date","mood(1-10)","sleep_hours","work_hours","exercise_mins","social_time","screen_time_hours","notes"])

# Ask if user wants to add a new entry
choice = input("Do you want to add today's data? (yes/no): ").strip().lower()
if choice == "yes":
    date = datetime.now().strftime("%Y-%m-%d")
    mood = int(input("Mood (1–10): "))
    sleep = float(input("Hours of sleep: "))
    work = float(input("Work/study hours: "))
    exercise = float(input("Minutes of exercise: "))
    social = float(input("Social time (hours): "))
    screen = float(input("Screen time (hours): "))
    notes = input("Notes or comments: ")

    new_row = {
        "date": date, "mood(1-10)": mood, "sleep_hours": sleep, "work_hours": work,
        "exercise_mins": exercise, "social_time": social, "screen_time_hours": screen, "notes": notes
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv("life_balance.csv", index=False)
    print("✅ Data saved successfully!\n")

# Convert types
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Compute insights
print("📈 Analyzing your balance data...\n")

correlations = df.corr(numeric_only=True)["mood(1-10)"].sort_values(ascending=False)
print("✨ Mood Correlations:")
print(correlations, "\n")

avg_sleep = df["sleep_hours"].mean()
avg_work = df["work_hours"].mean()
avg_screen = df["screen_time_hours"].mean()
print(f"🛌 Avg Sleep: {avg_sleep:.1f} hrs | 💼 Avg Work: {avg_work:.1f} hrs | 📱 Avg Screen Time: {avg_screen:.1f} hrs")

# Insights
if avg_sleep < 6:
    print("⚠️ You might be sleep-deprived. Your mood may improve with more rest.")
if avg_screen > 7:
    print("🔴 High screen time detected! Try reducing it for better focus.")
if correlations["sleep_hours"] > 0.3:
    print("💤 Sleep has a strong positive impact on your mood.")
if correlations["work_hours"] < -0.4:
    print("😩 Too much work time might be lowering your mood.")
if correlations["exercise_mins"] > 0.3:
    print("🏃 Regular exercise is improving your happiness!")

# Plot mood trend
plt.figure(figsize=(10,5))
plt.plot(df["date"], df["mood(1-10)"], marker='o', color='green')
plt.title("Mood Over Time")
plt.xlabel("Date")
plt.ylabel("Mood (1–10)")
plt.grid(True)
plt.show()
