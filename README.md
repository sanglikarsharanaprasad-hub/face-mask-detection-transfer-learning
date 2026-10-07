# Face Mask Detection Using Transfer Learning

A simple computer vision project that uses **MobileNetV2 transfer learning** to classify images as **Mask** or **No Mask** and performs real-time prediction using a webcam.

## Project Overview

This project demonstrates how a pretrained deep learning model can be adapted for a specific image-classification task.

Instead of training a neural network from scratch, we use **MobileNetV2 pretrained on ImageNet** and add a small classification layer for two classes:

* Mask
* No Mask

The trained model is then integrated with **OpenCV** for real-time webcam prediction.

## Technologies Used

* Python
* TensorFlow / Keras
* MobileNetV2
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn
* Google Colab

## Project Workflow

```text
Kaggle Dataset
      ↓
Dataset Cleaning & Preprocessing
      ↓
Train / Validation Split
      ↓
MobileNetV2 Transfer Learning
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Accuracy & Loss Graphs
      ↓
Confusion Matrix
      ↓
OpenCV Webcam
      ↓
Real-Time Mask / No Mask Prediction
```

## Project Structure

```text
face-mask-detection/
│
├── dataset/
│   └── README.md
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── webcam.py
│
├── notebooks/
│   └── training.ipynb
│
├── models/
│
├── results/
│   ├── accuracy.png
│   ├── loss.png
│   └── confusion_matrix.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

The dataset is obtained from Kaggle.

The original dataset is **not included in this GitHub repository** because of file size and dataset licensing considerations.

Download the dataset from the Kaggle source provided in:

```text
dataset/README.md
```

After downloading, follow the preprocessing instructions to create the required training and validation folders.

## Classes

The model performs binary classification:

```text
1. Mask
2. No Mask
```

## Model

We use **MobileNetV2 pretrained on ImageNet** as the base model.

The basic architecture is:

```text
Input Image (224 × 224 × 3)
          ↓
MobileNetV2
          ↓
Global Average Pooling
          ↓
Dropout
          ↓
Dense Layer
          ↓
Mask / No Mask
```

Transfer learning allows us to obtain useful image features from a pretrained model without training a deep neural network completely from scratch.

## Training

The dataset is divided approximately into:

* 80% Training
* 20% Validation

The images are resized to `224 × 224` during the TensorFlow preprocessing pipeline.

Simple data augmentation is used to improve generalization.

## Evaluation

The model is evaluated using:

* Validation accuracy
* Training accuracy
* Validation loss
* Training loss
* Confusion matrix
* Precision
* Recall
* F1-score

The generated graphs and evaluation results are stored in the `results/` directory.

## Real-Time Webcam Demo

After training, the saved model is connected to OpenCV.

The webcam captures frames and the model predicts:

```text
MASK
```

or

```text
NO MASK
```

along with the prediction confidence.

Press **Q** to close the webcam window.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Prepare the dataset

```bash
python src/preprocess.py
```

### 2. Train the model

Training can be performed using Google Colab or locally if suitable hardware is available.

### 3. Evaluate the model

```bash
python src/evaluate.py
```

### 4. Run the webcam demo

```bash
python src/webcam.py
```

## Team Contributions

| Member   | Contribution                                      |
| -------- | ------------------------------------------------- |
| Member 1 | Dataset collection, cleaning and preprocessing    |
| Member 2 | MobileNetV2 model development and training        |
| Member 3 | Model evaluation, graphs and performance analysis |
| Member 4 | OpenCV real-time webcam integration               |

## Future Improvements

Possible future improvements include:

* Face detection before classification
* Detection of multiple faces
* Improved dataset diversity
* Real-time bounding boxes
* Mobile or web deployment
* Improved performance under different lighting conditions

## Disclaimer

This project is developed for educational and demonstration purposes.
