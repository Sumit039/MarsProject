# MarsProject (AI-ML_audio_processing)-
## This project focuses on detecting emotions from speech using audio signals. The system classifies audio into emotions like happy, sad, angry, calm, disgust, surprised, fearful and neutral using machine learning and deep learning models.
# Project Description
The objective of this project is to detect human emotions from speech audio samples using a 1D Convolutional Neural Network (CNN). The model is trained and tested on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset, which contains recordings labeled with different emotional states.
The aim is to effectively detect and classify emotions like angry, calm, disgust, fearful, happy, neutral, sad, and surprised, which can be used in emotion-aware systems like mental health support systems, virtual assistants, or call center monitoring.
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
Used train-test split with stratified sampling to maintain class balance.
Features normalized using StandardScaler
#### Model Architecture: 1D CNN
Loss Function: Categorical Crossentropy
Optimizer: Adam
Metrics: Accuracy
Epochs: 100
### Evaluation Metrics:
##### Accuracy: (0.80)
The proposed 1D CNN model achieved an accuracy of 80%, demonstrating effective emotion classification from speech.

##### F1-Score (Macro): (0.79)
The macro-average F1-score of 0.79 indicates balanced performance across all emotion classes

##### F1-Score (per class): (0.80)
the weighted-average F1-score of 0.80 reflects strong overall classification quality, considering class distribution.

#### Classification Report

| Emotion   | Precision | Recall | F1-score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Angry     | 0.81      | 0.79   | 0.80     | 80      |
| Calm      | 0.80      | 0.65   | 0.72     | 37      |
| Disgust   | 0.75      | 0.84   | 0.79     | 74      |
| Fearful   | 0.79      | 0.88   | 0.83     | 67      |
| Happy     | 0.76      | 0.98   | 0.86     | 65      |
| Neutral   | 0.85      | 0.67   | 0.75     | 49      |
| Sad       | 0.93      | 0.71   | 0.81     | 79      |
| Surprised | 0.79      | 0.82   | 0.80     | 40      |

#### model correctly classified 80% of the total test samples with balanced performance across multiple emotion classes
