import os
import random
import time
from PIL import Image
import pandas as pd
import arabic_reshaper
from bidi.algorithm import get_display
import joblib


model = joblib.load("model_names_faces.pkl")

#funtion to display arabic words correctly (vs code runtime dosent support arabic)
def display_arabic(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)


#use file of faces
faces_folder = "RenamedFaces"
face_files = [f for f in os.listdir(faces_folder) if f.endswith(".png")]


# Show random face-name pairs for learning 
# (The number can be adjested based on the difficulty needed, and the child's preformance)
total_trials = 5
learning_faces = random.sample(face_files, total_trials)

print("LEARNING PHASE: Memorize the names and faces!")

for face in learning_faces:
    img_path = os.path.join(faces_folder, face)
    img = Image.open(img_path)
    img.show()
    
    name = face.split('.')[0]    #from the rpefex
    print(f" Name: {display_arabic(name)}")
    
    time.sleep(3)  # show for 3 seconds
    img.close()
    print("------")



print("\n \n \n \n \n \n \n \n \n")
print("\nTEST PHASE: Who is this? Type the correct name.")
correct_count = 0
reaction_times = []

for face in learning_faces:
    img_path = os.path.join(faces_folder, face)
    correct_name = face.split('.')[0]
    
    img = Image.open(img_path)
    img.show()
    
    # Wait 3 seconds
    time.sleep(3)

    # Close the image 
    os.system("taskkill /f /im Microsoft.Photos.exe >nul 2>&1")

    
    start = time.time()
    answer = input("Who is this? (Type the Arabic name): ").strip()  #time taken to answer
    end = time.time()
    
    img.close()
    reaction_time = end - start
    reaction_times.append(reaction_time)

    if answer == correct_name:  #check 
        print("Correct!")
        correct_count += 1
    else:
        print(f" Wrong. Correct answer: {display_arabic(correct_name)}")


# Calculate Scores
accuracy = correct_count / total_trials
avg_time = sum(reaction_times) / total_trials
time_score = max(0, 1 - avg_time) 
engagement_score = 0.85  # static for now 

# Total score and level
total_score = accuracy * 0.6 + time_score * 0.3 + engagement_score * 0.1

# Use trained model to predict level
input_features = pd.DataFrame([{
    "Accuracy_Score": accuracy,
    "Time_Score": time_score,
    "Engagement_Score": engagement_score
}])

performance_level = model.predict(input_features)[0]

# Show result
print("\nGame Over!")
print(f"Accuracy: {accuracy * 100:.1f}%")
print(f"Average Reaction Time: {avg_time:.2f} sec")
print(f"Total Score: {total_score:.2f}")
print(f"Performance Level: {performance_level}")
