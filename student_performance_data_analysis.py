import pandas as pd

# 1️⃣ Create DataFrame from dictionary
data = {
    'Name': ['Jelina', 'Pratik', 'Kushal', 'Aayusha', 'Saurav'],
    'Department': ['Computer', 'Computer', 'DevOps', 'Computer', 'Electrical'],
    'Age': [20, 22, 24, 19, 21],
    'Score': [88, 91, 85, 79, 90],
    'Attendance (%)': [95, 89, 92, 80, 88]
}

df = pd.DataFrame(data)

# 2️⃣ View basic info
print("----- Dataset Info -----")
df.info()
print("\n")

# 3️⃣ Display first few rows
print("----- First 3 Students -----")
print(df.head(3).to_string(index=False))
print("\n")

# 4️⃣ Basic statistics
print("----- Summary Statistics -----")
print(df.describe().round(2))
print("\n")

# 5️⃣ Accessing specific columns
print("----- Names of Students -----")
print(df['Name'].tolist())
print("\n")

# 6️⃣ Filter: Students with Score > 85
print("----- High Scorers (Score > 85) -----")
high_scorers = df[df['Score'] > 85].reset_index(drop=True)
print(high_scorers.to_string(index=False))
print("\n")

# 7️⃣ Add a new column (Pass/Fail)
df['Result'] = df['Score'].apply(lambda x: 'Pass' if x >= 80 else 'Fail')
print("----- Added 'Result' Column -----")
print(df.to_string(index=False))
print("\n")

# 8️⃣ Sort by Score descending
print("----- Sorted by Score (Descending) -----")
sorted_df = df.sort_values(by='Score', ascending=False, ignore_index=True)
print(sorted_df.to_string(index=False))
print("\n")

# 9️⃣ Remove a column
df.drop(columns=['Attendance (%)'], inplace=True)
print("----- After Dropping Attendance Column -----")
print(df.to_string(index=False))
