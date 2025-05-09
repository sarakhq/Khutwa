import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Display all rows/columns for better visualization
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.6f}'.format)

# Raw data input (synthetic for Delayed Visual Recognition)
np.random.seed(42)
participants = 30
trials_per_participant =5# 5 shapes to memorize per participant

# Raw trial data (no scoring here)
df = pd.DataFrame({
    "Subject_ID": np.repeat(np.arange(1, participants + 1), trials_per_participant),
    "Trial": np.tile(np.arange(1, trials_per_participant + 1), participants),
    "Shape": np.random.choice(["Square", "Triangle", "Circle"], size=participants * trials_per_participant),
    "Correct": np.random.choice([1, 0], size=participants * trials_per_participant, p=[0.7, 0.3]),
    "Reaction_Time_Avg": np.random.uniform(1.5, 15.0, size=participants * trials_per_participant),
    "Distractors_Selected": np.random.randint(0, 4, size=participants * trials_per_participant)
})

print("\nRaw Trial-Level Data (df):")
print(df.head())

# Grouped summary by subject
def compute_time_score(rt, max_time=15.0):
    return max(0, 1 - rt / max_time)

def get_level(score):
    if score >= 0.85: return 5
    elif score >= 0.65: return 4
    elif score >= 0.45: return 3
    elif score >= 0.25: return 2
    else: return 1

# Aggregate data by subject
summary_df = df.groupby("Subject_ID").agg({
    "Correct": "sum",
    "Reaction_Time_Avg": "mean",
    "Distractors_Selected": "sum"
}).reset_index()


# Calculate scores
summary_df["Accuracy_Score"] = summary_df["Correct"] / trials_per_participant
summary_df["Time_Score"] = summary_df["Reaction_Time_Avg"].apply(compute_time_score)
summary_df["Engagement_Score"] = np.random.uniform(0.5, 1.0, size=len(summary_df))


# Calculate total score
summary_df["Total_Score"] = (
    summary_df["Accuracy_Score"] * 0.6 +
    summary_df["Time_Score"] * 0.3 +
    summary_df["Engagement_Score"] * 0.1
)

# Determine performance level
summary_df["Performance_Level"] = summary_df["Total_Score"].apply(get_level)
summary_df["Advance"] = summary_df["Performance_Level"] >= 4

print("\nGrouped Subject-Level Summary (accuracy_summary)::")
print(summary_df.head())

# Train the model
features = ["Accuracy_Score", "Time_Score", "Engagement_Score"]
X = summary_df[features]
y = summary_df["Performance_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Export model to use it in game
joblib.dump(model, "model_visual_recognition.pkl")

# Training and test accuracy
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)
train_acc = accuracy_score(y_train, train_preds)*100
test_acc = accuracy_score(y_test, test_preds)*100

print(f"\nTraining Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")
