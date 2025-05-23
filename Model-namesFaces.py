import os
import shutil
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score

#Display all rows/columns when printing for better visualisation 
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.3f}'.format)


# Map Faces to Arabic Names
rename_map = {
    "Cindy": "أميرة", "Emma": "ليلى", "Brian": "مروان", "Emily": "حنين", "Dylan": "مالك",
    "Chris": "خالد", "Danielle": "دانة", "Danny": "داني", "Elizabeth": "هيفاء", "Joel": "جميل",
    "Sarah": "سارة", "Sofia": "صفاء", "Stephen": "سفيان", "Tony": "هالا", "Tina": "تالا",
    "Vanessa": "غلا", "Victor": "احمد", "Zoe": "لينا", "Sargio": "سامر", "James": "جاسم"
}
source_folder = "20Faces - Copy"
destination_folder = "RenamedFaces"
os.makedirs(destination_folder, exist_ok=True)
for filename in os.listdir(source_folder):
    prefix = filename.split('_')[0] #each origenal pic is labeled with format: name_XXX
    if prefix in rename_map:
        new_name = f"{rename_map[prefix]}.png"
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, new_name)
        if not os.path.exists(dst_path):
            shutil.copyfile(src_path, dst_path)


#Raw data input, following the MAS style
np.random.seed(42)
participants = 50
trials_per_series = 10
series_labels = ["A", "B"]

#Raw trial data (no scoring here)
df = pd.DataFrame({
    "Subject_ID": np.repeat(np.arange(1, participants + 1), trials_per_series * 2),
    "Series": np.tile(np.repeat(series_labels, trials_per_series), participants),
    "Trial": list(range(1, 11)) * 2 * participants,
    "Correct": np.random.choice([1, 0], size=participants * trials_per_series * 2, p=[0.75, 0.25]),
    "Reaction_Time_Avg": np.random.uniform(2.0, 8.0, size=participants * trials_per_series * 2),
    "Engagement_Score": np.random.uniform(0.5, 1.0, size=participants * trials_per_series * 2)
})

print("\nRaw Trial-Level Data (df):")
print(df.head())


#Grouped series summary (at the end of the series, calculate scores)
def compute_time_score(rt, max_time=8.0):
    return max(0, 1 - rt / max_time)

#set threshholds
def get_level(score):
    if score >= 0.85: return 5
    elif score >= 0.65: return 4
    elif score >= 0.45: return 3
    elif score >= 0.25: return 2
    else: return 1

accuracy_summary = df.groupby(["Subject_ID", "Series"]).agg({
    "Correct": "sum",
    "Reaction_Time_Avg": "mean",
    "Engagement_Score": "mean"
}).reset_index()

accuracy_summary.rename(columns={"Correct": "Correct_Count"}, inplace=True)
accuracy_summary["Accuracy_Score"] = accuracy_summary["Correct_Count"] / 10
accuracy_summary["Time_Score"] = accuracy_summary["Reaction_Time_Avg"].apply(compute_time_score)

#Formela
accuracy_summary["Total_Score"] = (
    accuracy_summary["Accuracy_Score"] * 0.6 +
    accuracy_summary["Time_Score"] * 0.3 +
    accuracy_summary["Engagement_Score"] * 0.1
)

accuracy_summary["Performance_Level"] = accuracy_summary["Total_Score"].apply(get_level)
accuracy_summary["Advance"] = accuracy_summary["Performance_Level"] >= 4 #Flag to go to next level

print("\nGrouped Series-Level Summary (accuracy_summary):")
print(accuracy_summary.head())


#Make copy of the deneraget Synthetic data
synthetic_df = accuracy_summary.copy()

#Load real session data (From the game)
real_df = pd.read_csv("game_sessions.csv")


#combine Both
cols = ["Accuracy_Score", "Time_Score", "Engagement_Score", "Performance_Level", "Total_Score"]
synthetic_clean = accuracy_summary[cols]
real_clean = real_df[cols]

combined_df = pd.concat([synthetic_clean, real_clean], ignore_index=True)



# Train the model
X = combined_df[["Accuracy_Score", "Time_Score", "Engagement_Score"]]
y = combined_df["Performance_Level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))


# Predictions to check over/under fit
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Accuracy scores
train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)
print(f"Training Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")



#export model to use it in game
joblib.dump(model, "model_names_faces.pkl")
