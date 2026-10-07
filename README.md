# Face Mask Detection Using Transfer Learning

A simple computer vision project that uses **MobileNetV2 Transfer Learning** to classify images as **Mask** or **No Mask** and performs real-time prediction using a webcam.

## Project Overview

This project demonstrates how a pretrained deep learning model can be adapted for a specific image-classification task.

Instead of training a neural network from scratch, we use **MobileNetV2 pretrained on ImageNet** and add a classification layer for two classes:

- Mask
- No Mask

The trained model is then integrated with **OpenCV** for real-time webcam prediction.

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Pandas
- Google Colab

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
│   └── README.md
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

The dataset is obtained from **Kaggle**.

The original dataset is not included in this GitHub repository because of file size and dataset licensing considerations.

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

Transfer learning allows us to use useful image features learned from a large dataset without training a deep neural network completely from scratch.

## Training

The dataset is divided approximately into:

- 80% Training
- 20% Validation

The images are resized to:

```text
224 × 224
```

during the TensorFlow preprocessing pipeline.

Data augmentation is used to improve model generalization.

Typical augmentation techniques include:

- Horizontal flipping
- Small rotations
- Zooming
- Image shifting

## Model Training Process

The training process consists of the following steps:

1. Load the dataset.
2. Clean and organize the images.
3. Preprocess the images.
4. Resize images to `224 × 224`.
5. Load the pretrained MobileNetV2 model.
6. Freeze the initial MobileNetV2 layers.
7. Add classification layers.
8. Compile the model.
9. Train the model using the training dataset.
10. Validate the model using the validation dataset.
11. Save the trained model.

## Evaluation

The model is evaluated using:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss
- Confusion matrix
- Precision
- Recall
- F1-score

The generated graphs and evaluation results are stored in the `results/` directory.

Example evaluation output:

```text
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
```

## Real-Time Webcam Demo

After training, the saved model is connected to **OpenCV**.

The webcam captures frames and the model predicts:

```text
MASK
```

or

```text
NO MASK
```

along with the prediction confidence.

Example:

```text
MASK - 96.45%
```

or:

```text
NO MASK - 91.23%
```

Press **Q** to close the webcam window.

## Installation

### 1. Clone the Repository

Replace the repository URL with your actual GitHub repository URL.

```bash
git clone <your-github-repository-url>
```

### 2. Open the Project Directory

```bash
cd face-mask-detection
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

## Requirements

The main Python libraries required for this project are:

```text
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
pandas
```

You can install them manually using:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn pandas
```

## Running the Project

### 1. Prepare the Dataset

First download and organize the Kaggle dataset according to the structure described in:

```text
dataset/README.md
```

Then run:

```bash
python src/preprocess.py
```

### 2. Train the Model

Training can be performed using **Google Colab** with GPU acceleration or locally if suitable hardware is available.

Run:

```bash
python src/train.py
```

After training, the trained model will be saved in the `models/` directory.

### 3. Evaluate the Model

Run:

```bash
python src/evaluate.py
```

This generates evaluation results such as:

```text
results/accuracy.png
results/loss.png
results/confusion_matrix.png
```

### 4. Run the Webcam Demo

After the model has been trained and saved, run:

```bash
python src/webcam.py
```

The webcam will open and detect whether the person is wearing a mask.

Press:

```text
Q
```

to exit the webcam window.

## Google Colab

Google Colab can be used for model training because it provides access to GPU resources.

Recommended workflow:

```text
Upload / Connect Dataset
        ↓
Install Dependencies
        ↓
Preprocess Dataset
        ↓
Load MobileNetV2
        ↓
Train Model
        ↓
