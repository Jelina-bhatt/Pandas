import random
import datetime
import os

# File to store fortune history
HISTORY_FILE = "fortune_history.txt"

# Fun fortunes
fortunes = [
    "Today will be full of surprises!",
    "You will meet someone who changes your life.",
    "A fun adventure awaits you.",
    "Something you lost will be found soon.",
    "Expect laughter and joy today.",
    "Challenges today will make you stronger.",
    "A new opportunity is heading your way.",
    "Your creativity will shine today."
]

# Motivational quotes
quotes = [
    "Believe in yourself!",
    "Every day is a fresh start.",
    "Good things come to those who hustle.",
    "Stay positive, work hard, make it happen."
]

# Function to generate lucky numbers
def generate_lucky_numbers():
    return [random.randint(1, 100) for _ in range(5)]

# Function to save fortune to history
def save_to_history(name, fortune, numbers):
    with open(HISTORY_FILE, "a") as file:
        file.write(f"{datetime.date.today()} - {name} - {fortune} - Lucky Numbers: {numbers}\n")

# Function to show previous fortunes
def show_history():
    if os.path.exists(HISTORY_FILE):
        print("\n📜 Your Fortune History:")
        with open(HISTORY_FILE, "r") as file:
            for line in file:
                print(line.strip())
    else:
        print("\nNo fortune history found. Start predicting today!")

# Main program
print("✨ Welcome to the Enhanced Digital Fortune Teller ✨")
name = input("Enter your name: ")
birthdate = input("Enter your birthdate (YYYY-MM-DD): ")

# Generate today's fortune
today_fortune = random.choice(fortunes)
lucky_numbers = generate_lucky_numbers()
quote = random.choice(quotes)

# Display fortune
print(f"\nHello {name}! Here is your fortune for {datetime.date.today()}:")
print(f"🔮 Fortune: {today_fortune}")
print(f"🎲 Lucky Numbers: {lucky_numbers}")
print(f"💡 Quote: {quote}")

# Save fortune
save_to_history(name, today_fortune, lucky_numbers)

# Ask if user wants to see history
see_history = input("\nDo you want to see your past fortunes? (yes/no): ").lower()
if see_history == "yes":
    show_history()

print("\nThank you for using the Digital Fortune Teller! Come back tomorrow for a new prediction. ✨")
