# MarsProject (AI-ML_audio_processing)-
## This project addresses speech-based emotion recognition by classifying audio signals into categories such as happy, sad, angry, calm, disgust, surprised, fearful, and neutral. Both machine learning and deep learning models were employed to perform the classification based on extracted audio features.
# Project Description
The objective of this project is to perform emotion recognition from speech audio using a 1D Convolutional Neural Network (CNN). The model was trained and evaluated on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset, which provides labeled audio recordings representing various emotional states.The aim is to effectively detect and classify emotions like angry, calm, disgust, fearful, happy, neutral, sad, and surprised, which can be used in emotion-aware systems like mental health support systems, virtual assistants, or call center monitoring.
# Preprocessing Methodology
#### Filename Metadata Extraction
Audio filenames are parsed to extract emotion ID and other metadata
#### Feature Extraction
Extracted MFCC (Mel Frequency Cepstral Coefficients) features.
40 MFCCs per frame were averaged over time to generate a 1D feature vector per sample.
Final input shape per sample: (174, 40) → Reshaped for 1D CNN input: (174, 40)
#### Label Encoding
Converted emotion labels to numeric form using LabelEncoder.
#### Data Splitting & Scaling
Applied stratified train-test split to maintain class balance across sets 
Features standardized using StandardScaler 
#### Model Architecture: 1D CNN
Loss Function: Categorical Crossentropy
Optimizer: Adam
Metrics: Accuracy
Epochs: 100
### Evaluation Metrics:
#### Accuracy: (0.80)
##### The proposed 1D CNN model achieved an accuracy of 80%, demonstrating effective emotion classification from speech.

#### F1-Score (Macro): (0.79)
##### The macro-average F1-score of 0.79 indicates balanced performance across all emotion classes

#### F1-Score (per class): (0.80)
##### the weighted-average F1-score of 0.80 reflects strong overall classification quality, considering class distribution.

#### Classification Report (on intense emotions)

| Emotion   | Precision | Recall | F1-score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Surprised | 0.79      | 0.82   | 0.80     | 40      |
| Disgust   | 0.75      | 0.84   | 0.79     | 74      |
| Fearful   | 0.79      | 0.88   | 0.83     | 67      |


#### Classification Report (on common emotions)

| Emotion   | Precision | Recall | F1-score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Happy     | 0.76      | 0.98   | 0.86     | 65      |
| Calm      | 0.80      | 0.65   | 0.72     | 37      |
| Neutral   | 0.85      | 0.67   | 0.75     | 49      |
| Sad       | 0.93      | 0.71   | 0.81     | 79      |
| Angry     | 0.81      | 0.79   | 0.80     | 80      |


#### model correctly classified 80% of the total test samples with balanced performance across multiple emotion classes.
#### Happy has the highest recall (0.98) — the model is excellent at catching most happy samples.
#### Sad has the best precision (0.93) — when the model says "Sad", it's almost always right.

