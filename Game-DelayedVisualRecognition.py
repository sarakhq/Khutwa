import os
import random
import time
from PIL import Image
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model_visual_recognition.pkl")

# Dataset folder paths
dataset_folder = "geometric shapes dataset"
shape_folders = ["Square", "Triangle", "Circle"]

# Collect all image files from the dataset
all_images = []
for shape in shape_folders:
    shape_folder_path = os.path.join(dataset_folder, shape)
    
    if os.path.exists(shape_folder_path):
        for i in range(1, 51):
            img_name = f"{shape}{i}.png"
            img_path = os.path.join(shape_folder_path, img_name)
            
            if os.path.exists(img_path):
                # Store as full path for easier use
                all_images.append(os.path.join(shape, img_name))

print(f"Found {len(all_images)} images in the dataset.")

# Show random shapes for learning (5 images)
total_trials = 5
learning_shapes = random.sample(all_images, total_trials)

print("LEARNING PHASE: Memorize these shapes!")

for shape in learning_shapes:
    img_path = os.path.join(dataset_folder, shape)
    img = Image.open(img_path)
    img.show()
    
    print(f"Showing: {shape}")
    
    time.sleep(3)  # show for 3 seconds
    img.close()
    print("------")

print("\n" * 10)  # Clear screen (somewhat)
print("Now wait for 30 seconds to memorize the shapes...")
time.sleep(30)

print("\nTEST PHASE: Which shape did you see?")
correct_count = 0
distractor_selected = 0
reaction_times = []

for target_shape in learning_shapes:
    # Get distractors (shapes not in learning set)
    distractors = [img for img in all_images if img not in learning_shapes]
    these_distractors = random.sample(distractors, 3)
    
    # Combine options and shuffle
    options = [target_shape] + these_distractors
    random.shuffle(options)
    
    # Find correct answer index
    correct_index = options.index(target_shape) + 1
    
    # Display all options
    for i, option in enumerate(options, 1):
        img_path = os.path.join(dataset_folder, option)
        img = Image.open(img_path)
        img.show()
        print(f"Option {i}: {option}")
    
    # Get user answer
    start = time.time()
    try:
        answer = int(input("Which shape did you see before? (1-4): ").strip())
    except ValueError:
        answer = 0  # Invalid input
    end = time.time()
    
    # Close all images
    for i in range(len(options)):
        try:
            img.close()
        except:
            pass
    
    # Process result
    reaction_time = end - start
    reaction_times.append(reaction_time)
    
    if answer == correct_index:
        print("Correct!")
        correct_count += 1
    else:
        print(f"Wrong. Correct answer was option {correct_index}")
        if 1 <= answer <= 4:
            distractor_selected += 1

# Calculate Scores
accuracy = correct_count / total_trials
avg_time = sum(reaction_times) / len(reaction_times)
time_score = max(0, 1 - avg_time/15.0)  # Normalize time (max 15 seconds)
engagement_score = 0.85  # Static for now

# Total score calculation
total_score = accuracy * 0.6 + time_score * 0.3 + engagement_score * 0.1

# Use trained model to predict level
input_features = pd.DataFrame([{
    "accuracy_score": accuracy,
    "time_score": time_score,
    "engagement_score": engagement_score
}])

performance_level = model.predict(input_features)[0]

# Show results
print("\nGame Over!")
print(f"Correct Answers: {correct_count} out of {total_trials}")
print(f"Accuracy: {accuracy * 100:.1f}%")
print(f"Average Reaction Time: {avg_time:.2f} sec")
print(f"Total Score: {total_score:.2f}")
print(f"Performance Level: {performance_level}")
print(f"Advance to Next Level: {'Yes' if performance_level >= 4 else 'No'}")