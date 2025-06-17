# Age Detection CNN with Grad-CAM (Regression)

This repository implements a ResNet18-based regression model for predicting numerical ages, with Grad-CAM visualizations and classification metrics via age binning.

## Project Structure

age_detection_cnn/ ├── requirements.txt ├── age_detection_cnn.ipynb ├── model.py ├── weights/ │ └── resnet18_age_regression.pth ├── data/ │ └── UTKFace/ (placeholder; not included) ├── README.md


## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt





Download the UTKFace dataset to C:\Users\ponna\Downloads\drive-download-20250525T175035Z-1-001 or update data_dir in age_detection_cnn.ipynb.



Download model weights: Google Drive Link



Run age_detection_cnn.ipynb in Jupyter Notebook.

Dataset





Expected: UTKFace dataset with images named age_gender_race_date.jpg.



Update data loading in age_detection_cnn.ipynb if using a different structure.

Metrics





Regression: Mean Absolute Error (MAE, target <5 years), Mean Squared Error (MSE).



Classification: Accuracy, precision, recall, confusion matrix (ages binned into 0-20, 21-40, 41-60, 61+).



Visualizations: Grad-CAM heatmaps, confusion matrix, predicted vs. actual age plots.

Usage





Train and evaluate: Run all cells in age_detection_cnn.ipynb.



Predict ages: Last cell predicts numerical ages for three images with Grad-CAM.



Model weights saved to weights/resnet18_age_regression.pth.
