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
print(df.info(), "\n")

# 3️⃣ Display first few rows
print("----- First 3 Students -----")
print(df.head(3), "\n")

# 4️⃣ Basic statistics
print("----- Summary Statistics -----")
print(df.describe(), "\n")

# 5️⃣ Accessing specific columns
print("----- Names of Students -----")
print(df['Name'].to_list(), "\n")

# 6️⃣ Filter: Students with score > 85
print("----- High Scorers (Score > 85) -----")
high_scorers = df[df['Score'] > 85]
print(high_scorers, "\n")

# 7️⃣ Add a new column (Pass/Fail)
df['Result'] = ['Pass' if x >= 80 else 'Fail' for x in df['Score']]
print("----- Added 'Result' Column -----")
print(df, "\n")

# 8️⃣ Sort by Score descending
print("----- Sorted by Score (Descending) -----")
print(df.sort_values(by='Score', ascending=False), "\n")

# 9️⃣ Remove a column
df.drop('Attendance (%)', axis=1, inplace=True)
print("----- After Dropping Attendance Column -----")
print(df)
