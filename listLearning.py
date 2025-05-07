
# formela: total_score = accuracy_score * 0.6 + time_score * 0.3 + engagement_score * 0.1


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("list_learning_dataset (1).csv")
print(df.head())

# Function to compute MAS-style accuracy score
def compute_accuracy_score(row, max_recall=12, max_intrusions=5, clustering_bonus_weight=0.1):
    recall = row['Recall_Count']
    intrusions = row['Intrusion_Count']
    clustering = row['Clustering_Score']

    recall_score = recall / max_recall
    intrusion_penalty = min(intrusions / max_intrusions, 1)
    clustering_bonus = clustering * clustering_bonus_weight

    accuracy = recall_score - intrusion_penalty + clustering_bonus
    return max(min(accuracy, 1), 0)

# Apply accuracy score 
df['accuracy_score'] = df.apply(compute_accuracy_score, axis=1)

# Compute total score (only accuracy score is used here)
df['total_score'] = df['accuracy_score'] * 0.6  # rest of formula is : total_score = accuracy_score * 0.6 + time_score * 0.3 + engagement_score * 0.1

# Function to compute performance level based on total_score
def compute_performance_level_by_total_score(row):
    score = row['total_score']
    if score >= 0.51:
        return 5
    elif score >= 0.42:
        return 4
    elif score >= 0.33:
        return 3
    elif score >= 0.24:
        return 2
    else:
        return 1


print(df.head())


# Apply performance level
df['performance_level'] = df.apply(compute_performance_level_by_total_score, axis=1)

#Flag for going to the next level
df['advance'] = df['performance_level'] >= 4



print(df.head())


# Features and labels
X = df[['Recall_Count', 'Intrusion_Count', 'Clustering_Score']]
y = df['performance_level']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
