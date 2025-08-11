🧠 U-Net Based Image Segmentation Project
📌 Overview
This project implements a U-Net deep learning model for image segmentation, allowing pixel-level classification of input images.
It was developed and tested in Google Colab with dataset handling, training, evaluation, and visualization steps clearly documented.

📂 Project Structure
Data Loading & Preprocessing

Mount Google Drive to access datasets

Load images and corresponding masks

Resize and normalize data

Model Architecture

U-Net architecture implemented using TensorFlow/Keras

Encoder–Decoder structure with skip connections

Training

Compile model with appropriate optimizer and loss function

Train on dataset with validation split

Evaluation

Generate predictions

Convert predictions and ground truth to binary masks

Compute accuracy metrics and confusion matrix

Visualization

Display original image, ground truth mask, and predicted mask

Save results for further analysis

🛠️ Requirements
To run this notebook, install the following:

bash
Copy
Edit
pip install tensorflow numpy opencv-python matplotlib scikit-learn
🚀 How to Run
Open the notebook in Google Colab.

Upload the dataset to Google Drive and adjust paths in the notebook.

Run all cells in sequence to:

Preprocess data

Train the U-Net model

Evaluate and visualize results

📊 Evaluation Metrics
Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

Accuracy per class

📌 Key Functions
visualize_results(X_test, Y_test, preds, num_samples=15)
Displays comparison of input images, ground truth, and predictions.

Binary thresholding to convert predictions into 0–1 masks for evaluation.

📷 Example Output
Input Image	Ground Truth	Predicted Mask
