import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample training data for student performance
# Features: Hours Studied (per week), Attendance (%)
# Label: Exam Score (%)
data = {
    'hours': [5, 10, 15, 20, 2, 8, 12, 18, 22, 6],
    'attendance': [60, 75, 85, 95, 50, 70, 80, 90, 98, 65],
    'score': [55, 70, 82, 95, 45, 68, 78, 88, 98, 60]
}

# Create dataframe
df = pd.DataFrame(data)

# Features and labels
X = df[['hours', 'attendance']]
y = df['score']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'study_model.pkl')

print("Study model trained and saved successfully!")
