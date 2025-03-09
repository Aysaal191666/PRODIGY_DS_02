# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset directly 
data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(data_url)

# Display basic dataset info
print("📌 First 5 rows of the dataset:")
print(df.head())

print("\n📌 Dataset Info:")
print(df.info())

print("\n📌 Summary Statistics:")
print(df.describe())

# Checking for missing values
print("\n📌 Missing Values:")
print(df.isnull().sum())

# Handling missing values
df['Age'].fillna(df['Age'].median(), inplace=True)  # Fill missing Age with median
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Fill missing Embarked with mode
df.drop(columns=['Cabin'], inplace=True)  # Drop Cabin due to too many missing values

#  📊 Improved Data Visualization

# Set style for all plots
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))

# 📌 1. Survival Count
plt.subplot(2, 2, 1)
sns.countplot(x='Survived', data=df, palette=['#FF6F61', '#6B8E23'])
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")

# 📌 2. Passenger Class Distribution
plt.subplot(2, 2, 2)
sns.countplot(x='Pclass', data=df, palette='Blues_r')
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class (1st, 2nd, 3rd)")
plt.ylabel("Count")

# 📌 3. Survival Rate by Passenger Class
plt.subplot(2, 2, 3)
sns.barplot(x='Pclass', y='Survived', data=df, palette='coolwarm')
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

# 📌 4. Survival Rate by Gender
plt.subplot(2, 2, 4)
sns.barplot(x='Sex', y='Survived', data=df, palette=['#3498db', '#e74c3c'])
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")

plt.tight_layout()
plt.show()

# 📌 5. Age Distribution (Separate Plot)
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=30, kde=True, color='purple')
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Save cleaned dataset
df.to_csv("cleaned_titanic_data.csv", index=False)
print("\n✅ Data cleaning & EDA completed. Cleaned dataset saved as 'cleaned_titanic_data.csv'.")
