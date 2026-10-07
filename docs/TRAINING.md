# Training Documentation

## Overview

This document records the training configuration and the recorded training results for the face-mask detection model.

The information below is based on the existing `src/train.py`, `results/history.json`, and `results/validation_metrics.json`. No retraining is required to document these results.

## Dataset

The project uses the Kaggle Face Mask Dataset.

The dataset contains:

- `with_mask`: 3,725 images
- `without_mask`: 3,828 images
- Total: 7,553 images

The dataset is divided into:

- 80% training data
- 20% validation data

The split uses seed `42` so that the training and validation split is reproducible.

The original dataset images are downloaded separately and are not stored in Git.

Expected local dataset structure:

```text
data/
├── with_mask/
└── without_mask/
```

## Preprocessing

The training script uses the following preprocessing configuration:

- Image size: `160 × 160` pixels
- Color format: RGB
- Batch size: `32`
- Validation split: `20%`
- Random seed: `42`
- Class names:
  - `with_mask`
  - `without_mask`

The model contains a rescaling layer:

```text
Rescaling(1/127.5, offset=-1)
```

This converts pixel values from the range `[0, 255]` to approximately `[-1, 1]`.

The same preprocessing is stored inside the model so that webcam input can use the same preprocessing during prediction.

## Model Architecture

The project uses transfer learning with MobileNetV2.

The architecture is:

```text
Input (160 × 160 × 3)
        ↓
Rescaling
        ↓
MobileNetV2 (ImageNet pretrained)
        ↓
Global Average Pooling 2D
        ↓
Dropout (0.2)
        ↓
Dense (1 neuron, Sigmoid)
        ↓
Output
```

The MobileNetV2 base model is pretrained on ImageNet and is frozen during training:

```text
base.trainable = False
```

The final Dense layer uses a sigmoid activation for binary classification.

## Training Configuration

The training configuration from `src/train.py` is:

| Parameter | Value |
|---|---|
| Model | MobileNetV2 |
| Pretrained weights | ImageNet |
| Base model | Frozen |
| Input size | 160 × 160 × 3 |
| Batch size | 32 |
| Epochs | 5 |
| Validation split | 20% |
| Random seed | 42 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Binary Crossentropy |
| Metric | Accuracy |
| Dropout | 0.2 |
| Checkpoint monitoring | Validation loss |
| Save best model | Yes |

The best model is saved as:

```text
models/mask_detector.keras
```

The checkpoint uses `val_loss` and `save_best_only=True`, so the model with the lowest validation loss is retained.

## Recorded Training Results

The following values are recorded in `results/history.json`.

| Epoch | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
|---:|---:|---:|---:|---:|
| 1 | 94.46% | 0.14721 | 98.87% | 0.04837 |
| 2 | 98.73% | 0.04284 | 98.81% | 0.03510 |
| 3 | 99.17% | 0.03235 | 98.94% | 0.03062 |
| 4 | 99.24% | 0.02564 | 99.01% | 0.02889 |
| 5 | 99.32% | 0.02348 | 98.94% | 0.02800 |

## Validation Accuracy and Saved Checkpoint

Epoch 4 achieved the highest recorded validation accuracy:

```text
99.01%
```

However, the checkpoint is selected using validation loss rather than validation accuracy.

Epoch 5 produced a lower validation loss:

```text
0.02800
```

compared with epoch 4:

```text
0.02889
```

Therefore, the epoch 5 model is retained as the best checkpoint.

The recorded saved-model validation metrics are:

```text
Validation loss:     0.0279984474
Validation accuracy: 0.9894039631
```

The validation accuracy is approximately:

```text
98.94%
```

This value represents performance on the project's validation split. It should not be interpreted as an independent test-set accuracy or as guaranteed real-world webcam accuracy.

## Output Files

The training script generates the following files:

```text
models/
├── mask_detector.keras
└── class_names.json

results/
├── accuracy.png
├── loss.png
├── history.json
└── validation_metrics.json
```

### `models/mask_detector.keras`

Saved best-performing model checkpoint based on validation loss.

### `models/class_names.json`

Stores the class names:

```text
with_mask
without_mask
```

### `results/history.json`

Stores the training and validation accuracy and loss for all five epochs.

### `results/validation_metrics.json`

Stores the validation loss and accuracy of the saved model.

### `results/accuracy.png`

Plot of training and validation accuracy across the five epochs.

### `results/loss.png`

Plot of training and validation loss across the five epochs.

## Training on Windows

From the repository root, create a Python 3.11 virtual environment:

```text
py -3.11 -m venv .venv
```

Install the required packages:

```text
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Make sure the dataset is available under the `data/` directory, then run:

```text
.venv\Scripts\python.exe src\train.py
```

The first run downloads the pretrained ImageNet weights for MobileNetV2.

## Reproducibility

The training script sets the TensorFlow random seed to `42` and uses the same seed for the training and validation dataset split.

The recorded results in this document come from the existing training history and validation metrics files. Retraining is not required to use or document the existing model.

## Verification

The training configuration was checked against `src/train.py`, and the recorded five-epoch results were checked against:

- `results/history.json`
- `results/validation_metrics.json`

The documentation does not claim that the model was retrained as part of this documentation change.
