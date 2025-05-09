#importing libraries:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


#to display full rows and col when printing:
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.6f}'.format)
pd.set_option('display.width', 100)



# Load dataset
df = pd.read_csv("list_learning_dataset (1).csv")
#explaination of features: 
#


# Simulate realistic reaction times (in seconds)
# assuming most kids take between 1.5 to 15 seconds per item
np.random.seed(42)
df['Reaction_Time_Avg'] = np.random.uniform(low=1.5, high=15.0, size=len(df))


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


# Add time_score (normalize so lower time > higher score)
def compute_time_score(row, max_time=15.0):
    rt = row['Reaction_Time_Avg']
    return max(0, 1 - rt / max_time)

df['time_score'] = df.apply(compute_time_score, axis=1)

# Add dummy engagement_score for now (making model for facial reconition)
df['engagement_score'] = np.random.uniform(low=0.5, high=1.0, size=len(df))



# Total score with full formula:  total_score = accuracy_score * 0.6 + time_score * 0.3 + engagement_score * 0.1
df['total_score'] = (
    df['accuracy_score'] * 0.6 +
    df['time_score'] * 0.3 +
    df['engagement_score'] * 0.1
)


# Function to compute performance level based on total_score (Thresholds)
def compute_performance_level_by_total_score(row):
    score = row['total_score']
    if score >= 0.85:
        return 5
    elif score >= 0.7:
        return 4
    elif score >= 0.55:
        return 3
    elif score >= 0.4:
        return 2
    else:
        return 1



# Apply performance level
df['performance_level'] = df.apply(compute_performance_level_by_total_score, axis=1)

#Flag for going to the next level
df['advance'] = df['performance_level'] >= 4

#see sdded coloumns to data
df = df.drop(["Word_List", "Recalled_Words","Intrusions"], axis=1)
print(df.head())




#Features and labels
X = df[['accuracy_score', 'time_score', 'engagement_score']]
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



#---------------test on new data:

#RAW input from the game: example
recall = 10
intrusions = 1
clustering = 0.6
reaction_time = 2.5
engagement = 0.75  # until model is implemented

#Convert raw input to model features
def compute_accuracy_score_from_raw(recall, intrusions, clustering, max_recall=12):
    recall_score = recall / max_recall
    intrusion_penalty = min(intrusions / 5, 1)
    clustering_bonus = clustering * 0.1
    return max(min(recall_score - intrusion_penalty + clustering_bonus, 1), 0)

def compute_time_score_from_raw(rt, max_time=5.0):
    return max(0, 1 - rt / max_time)

accuracy = compute_accuracy_score_from_raw(recall, intrusions, clustering)
time_score = compute_time_score_from_raw(reaction_time)

#  Input to the model
X_new = pd.DataFrame([{
    'accuracy_score': accuracy,
    'time_score': time_score,
    'engagement_score': engagement
}])


#Predict using model
predicted_level = model.predict(X_new)[0]
print("Example on unseen Data: Predicted Performance Level:", predicted_level)



#plotting to see some insights
features = ['accuracy_score', 'time_score', 'engagement_score']
# for feature in features:
#     plt.figure(figsize=(6, 4))
#     sns.boxplot(x='performance_level', y=feature, data=df)
#     plt.title(f"{feature} vs Performance Level")
#     plt.show()

# plt.figure(figsize=(6, 4))
# sns.boxplot(x='performance_level', y='accuracy_score', data=df)
# plt.title("Accuracy Score vs Performance Level")
# plt.xlabel("Performance Level")
# plt.ylabel("Accuracy Score")
# plt.show()

# #corr between features

# plt.figure(figsize=(8, 6))
# sns.heatmap(df[features + ['performance_level']].corr(), annot=True, cmap='coolwarm')
# plt.title("Correlation Heatmap")
# plt.show()


# sns.lmplot(x='performance_level', y='accuracy_score', data=df)
# plt.title("Linear Relationship Between Performance Level and Accuracy Score")
# plt.xlabel("Performance Level")
# plt.ylabel("Accuracy Score")
# plt.show()


train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Accuracy scores
train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

print(f"Training Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

