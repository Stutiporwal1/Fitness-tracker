# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/gym_members_exercise_tracking_synthetic_data.csv")
df.head()

print(df.shape)

print(df.columns)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Weight (kg)'] = df['Weight (kg)'].fillna(df['Weight (kg)'].median())
df['Height (m)'] = df['Height (m)'].fillna(df['Height (m)'].mean())

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Workout_Type'] = df['Workout_Type'].fillna(df['Workout_Type'].mode()[0])

print("Remaining missing values per column:")
print(df.isnull().sum())

duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

df = df.drop_duplicates()
print("Duplicates removed.")

print("Unique values in Max_BPM before cleaning:", df['Max_BPM'].unique())

df['Max_BPM'] = pd.to_numeric(df['Max_BPM'], errors='coerce')

df['Age'].hist(bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

df['Calories_Burned'].hist(bins=20)
plt.title("Calories Burned Distribution")
plt.xlabel("Calories Burned")
plt.ylabel("Frequency")
plt.show()

sns.countplot(x='Gender', data=df)
plt.title("Gender Distribution")
plt.show()

sns.countplot(x='Workout_Type', data=df)
plt.title("Workout Type Distribution")
plt.show()

bmi_trend = df.groupby('Workout_Frequency (days/week)')['BMI'].mean().reset_index()
print(bmi_trend)

correlation = df['Session_Duration (hours)'].corr(df['Calories_Burned'])
print(f"Correlation between Session_Duration and Calories_Burned: {correlation:.2f}")

sns.scatterplot(x='Session_Duration (hours)', y='Calories_Burned', hue='Workout_Type', data=df)
plt.title("Session Duration vs Calories Burned")
plt.xlabel("Session Duration (hours)")
plt.ylabel("Calories Burned")
plt.show()

sns.lineplot(data=bmi_trend, x='Workout_Frequency (days/week)', y='BMI')
plt.title("BMI vs Workout Frequency")
plt.xlabel("Workout Frequency (days/week)")
plt.ylabel("Average BMI")
plt.show()


