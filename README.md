# Khutwa (خطوة) - AI-Based Adaptive Memory Assessment

LOGO PIC HERE 

## Overview

Khutwa (meaning "Step" in Arabic) is an innovative tech initiative under the Kafak program, aimed at rehabilitating child soldiers in Yemen. The platform provides psychological support and cognitive development through interactive, AI-powered games that adapt to each child's unique abilities and needs.

This repository contains the memory assessment components of Khutwa, featuring games inspired by the Memory Assessment Scale (MAS) to evaluate and enhance various types of memory:

- **Short-term Memory**: Enhances immediate recall skills
- **Working Memory**: Supports problem-solving and mental manipulation
- **Long-term Memory**: Aids in retaining information over extended periods

## Project Structure

```
khutwa/
│── Game-DelayedVisualRecognition.py  # Long-term visual memory assessment
│── Game-namesFaces.py               # Working memory for name-face associations
│─ Model-delayedVisualRecognition.py # Visual recognition model training
│── Model-namesFaces.py              # Names-faces association model training
│── Model-listLearning.py            # List learning model training
├─ geometric shapes dataset/        # Visual shapes for recognition tasks
│   ├── Square/
│   ├── Triangle/
│   └── Circle/
├── RenamedFaces/                    # Face images with Arabic names
│── list_learning_dataset.csv        # Verbal memory dataset
├
├── game_sessions.csv                # Names-faces game session data
│── Shapes_game_sessions.csv         # Shape recognition session data
└── requirements.txt                     # Dependencies
```

## Memory Assessment Games

### 1. Delayed Visual Recognition

This game assesses long-term visual memory by testing a child's ability to recognize previously seen geometric shapes.

GAMEPLAY PIC HERE 

**Game Flow:**
1. **Learning Phase**: The child is shown 5 random shapes for memorization (5 seconds each)
2. **Delay Period**: A 30-second waiting period
3. **Test Phase**: Each memorized shape is presented alongside 3 distractors
4. **Scoring**: Performance is measured by accuracy, reaction time, and engagement

### 2. Names-Faces Association

This game measures working memory by testing a child's ability to associate names with faces.

GAMEPLAY PIC HERE 

**Game Flow:**
1. **Learning Phase**: The child is shown 5 faces with their corresponding Arabic names
2. **Test Phase**: The faces are shown again, and the child must recall the associated name
3. **Scoring**: Performance is evaluated based on recall accuracy and response time

### 3. List Learning

This game assesses verbal memory through auditory learning tasks.

**Game Flow:**
1. **Learning Phase**: The child listens to a list of words from different semantic categories
2. **Test Phase**: The child must recall as many words as possible
3. **Scoring**: Performance is evaluated based on correct recalls, intrusions, and clustering

## AI Performance Assessment Model

Khutwa uses a RandomForest classifier to predict a child's performance level based on three key metrics:

1. **Accuracy Score (60%)**: Measures correct responses in games
2. **Time Score (30%)**: Evaluates reaction time and processing speed
3. **Engagement Score (10%)**: Analyzes facial expressions and attention levels

Performance is classified into 5 levels:
- **Level 5**: Excellent (Score ≥ 0.85)
- **Level 4**: Above Average (Score ≥ 0.7)
- **Level 3**: Average (Score ≥ 0.55)
- **Level 2**: Below Average (Score ≥ 0.4)
- **Level 1**: Needs Improvement (Score < 0.4)

The system automatically adjusts difficulty based on performance, with children scoring at Level 4 or above advancing to more challenging tasks.

## Continuous Learning System

Game sessions are automatically saved to CSV files, allowing the model to continuously learn and improve over time:

```python
session_data = {
    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Accuracy_Score": accuracy,
    "Time_Score": time_score,
    "Engagement_Score": engagement_score,
    "Total_Score": total_score,
    "Performance_Level": performance_level
}
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- PIL (Pillow)
- pandas
- numpy
- scikit-learn
- joblib
- arabic-reshaper (for Names-Faces game)
- python-bidi (for Names-Faces game)

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/khutwa.git
   cd khutwa
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the datasets:
   - Unzip  the "geometric shapes dataset" folder for shape images 
   - Ensure face images are in the "RenamedFaces" folder

4. Train the models:
   ```
   python models/Model-delayedVisualRecognition.py
   python models/Model-namesFaces.py
   python models/Model-listLearning.py
   ```

5. Run the games:
   ```
   python games/Game-DelayedVisualRecognition.py
   python games/Game-namesFaces.py
   ```

## Project Contributors

- Sarah Alkahwaji 222410358
- Aljawharah Aljubair 222410187
- Prince Sultan University
- Instructor: Dr. Heyfa Ammar

## Acknowledgements

This project is part of an initiative to support children affected by conflict in Yemen. It draws inspiration from clinical memory assessment techniques and applies them in an engaging, therapeutic context.

## License

[MIT License](LICENSE)
