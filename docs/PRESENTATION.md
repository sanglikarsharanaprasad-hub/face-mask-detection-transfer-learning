# MaskLab Presentation and Demo Guide

## 1. Project introduction

Our project is Face Mask Detection using Transfer Learning.

It classifies detected faces into two categories:
- Mask
- No Mask

We trained a MobileNetV2 classifier and created two demonstrations:
an OpenCV webcam application and a browser website.

## 2. Problem statement

Training a deep learning model from scratch requires significant
data, computing resources and time.

This project uses a pretrained model to build a working image
classifier within a small number of training epochs.

The final deliverables include:
- A trained Keras model.
- Training and validation accuracy plots.
- Training and validation loss plots.
- A real-time webcam demonstration.
- A browser interface for detection.

The implemented scope is face masks. Helmet detection is future work.

## 3. Transfer learning

Transfer learning reuses features learned by an existing model.

MobileNetV2 was pretrained on ImageNet. Its convolutional layers
provide image features that can be reused for mask classification.

We froze the pretrained backbone and trained a small classification
head on our dataset.

This reduced the number of parameters being trained.

## 4. Dataset

The project used the Kaggle face mask dataset.

Dataset page:
https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

The cleaned dataset contained:
- 3,725 images with masks.
- 3,828 images without masks.
- 7,553 images in total.

The split contained:
- 6,043 training images.
- 1,510 validation images.

The dataset images are downloaded separately and excluded from Git.
Download and usage conditions should be checked on the dataset page.

## 5. Model architecture

The training pipeline uses:
1. RGB input images resized to 160 by 160 pixels.
2. Rescaling from pixel values to the range -1 to 1.
3. A frozen MobileNetV2 backbone.
4. Global average pooling.
5. Dropout with rate 0.2.
6. A dense layer with one sigmoid output.

The sigmoid output represents the No Mask class.

The class order is:
- Class 0: with_mask
- Class 1: without_mask

At the classification threshold of 0.5:
- A value below 0.5 produces Mask.
- A value of 0.5 or above produces No Mask.

## 6. Training settings

- Image size: 160 by 160 pixels.
- Batch size: 32.
- Epochs: 5.
- Optimizer: Adam.
- Learning rate: 0.001.
- Loss: binary cross-entropy.
- Reported metric: accuracy.
- Split seed: 42.

The best checkpoint was selected using the lowest validation loss.

## 7. Results

The saved model achieved approximately 98.94% validation accuracy.

Epoch 4 had a higher validation accuracy of approximately 99.01%.
However, epoch 5 had the lowest validation loss and was selected
as the saved checkpoint.

Validation accuracy describes performance on the validation split.
It is not an independent test-set result or a guarantee of accuracy
on every live camera frame.

Show these files during the presentation:
- results/accuracy.png
- results/loss.png
- results/history.json
- results/validation_metrics.json

## 8. Python webcam workflow

The OpenCV application:
1. Opens the webcam.
2. Detects faces using a Haar cascade.
3. Crops each detected face.
4. Converts the crop from BGR to RGB.
5. Resizes it to the classifier input size.
6. Runs the Keras model.
7. Draws the predicted label and confidence.

The Keras model includes its normalization layer.

Controls:
- S saves a demonstration screenshot.
- Q closes the webcam window.

## 9. Browser workflow

The website:
1. Requests camera permission.
2. Loads the bundled JavaScript libraries and models.
3. Detects faces using BlazeFace.
4. Crops and resizes each detected face.
5. Normalizes the input for the converted classifier.
6. Runs classification using TensorFlow.js.
7. Displays the label and confidence.

The browser implementation runs inference in the browser.
It does not require a Flask inference server.

The browser and Python demonstrations use different face detectors.
Their face crops and predictions can therefore differ.

## 10. Vishnu's completed contribution

My completed contribution includes:
- Deployment documentation.
- A README link to the deployment guide.
- Manual browser verification on macOS.
- A Python static deployment checker.
- Documentation of checker usage and limitations.

The checker verifies required assets, model manifests and declared
weight files before deployment.

It uses the Python standard library and does not modify project files.

Evidence includes commits and merged pull requests under my GitHub
account. Other members should describe their own completed work.

This presentation guide is an additional contribution once reviewed
and merged.

## 11. Automated checks demonstrated

Run from the repository root:

```bash
python3 src/check_web_deployment.py
```

The existing website passed with 12 nonempty files.

A nonexistent website directory was also tested.
It produced a failure message and exit code 1.

The checker detects:
- Missing files.
- Empty files.
- Unreadable files.
- Invalid model manifest JSON.
- Missing declared weight shards.
- Git LFS pointer files.
- Weight paths outside the website directory.

These checks do not establish that the model weights are correct.
Model loading and prediction behavior require browser testing.

## 12. Suggested five-minute presentation

### First minute: introduction

Explain the problem, the two classes and the project deliverables.

### Second minute: model

Explain transfer learning, MobileNetV2 and the classification head.

### Third minute: results

Show the accuracy and loss plots.
Explain the saved validation result and checkpoint selection.

### Fourth minute: live demonstration

Start the browser camera.
Demonstrate Mask and No Mask predictions.
Stop the camera.

### Fifth minute: contribution and limitations

Run the deployment checker.
Explain your completed contribution and show merged PR evidence.
Discuss limitations and possible improvements.

## 13. Local browser demo

From the repository root:

```bash
python3 -m http.server 8000 --directory web
```

Open:
http://localhost:8000

Allow camera access and wait for model loading.

Use a well-lit scene with the face clearly visible.
Demonstrate both categories and the Stop Camera control.

Press Control + C in the terminal to stop the local server.

Local browser functionality was verified on macOS on 7 October 2026.

## 14. Demo preparation checklist

Before the exam:
- Confirm the repository is on the intended submission branch.
- Run the static deployment checker.
- Open the website and check model loading.
- Check browser camera permission.
- Keep a mask ready for the demonstration.
- Open the accuracy and loss plots.
- Keep the merged PR links available.
- Prepare screenshots in case live camera access fails.

Do not describe a screenshot demonstration as a live test.

## 15. Deployment status

The website is designed for static hosting over HTTPS.

The web directory contains the page, scripts, libraries and models.
These files must remain together with their relative paths preserved.

Vercel deployment is not yet verified in this guide.
Only claim a public deployment works after testing its actual URL.

See DEPLOYMENT.md for setup and troubleshooting.

## 16. Limitations

- Face detection can fail at unusual angles or distances.
- Poor lighting can affect both detection and classification.
- Model confidence is not a guarantee of correctness.
- The validation split is not a separate independent test dataset.
- Face mask classification does not assess medical protection.
- Helmet classification is not implemented.
- Performance can vary across cameras and browsers.

## 17. Future improvements

Possible future work includes:
- Testing on an independent dataset.
- Reporting precision, recall and a confusion matrix.
- Improving robustness to lighting and face angle.
- Measuring browser inference performance.
- Training a separate helmet classifier.
- Verifying a public HTTPS deployment.

These are proposals, not completed project results.

## 18. Viva questions and answers

### Why did you use transfer learning?

It lets us reuse pretrained image features and train a smaller
classification head with less time and computing effort.

### Why MobileNetV2?

It is a compact image feature extractor suitable for a project
that needs webcam and browser demonstrations.

### What does freezing the backbone mean?

The pretrained backbone weights are not updated during training.
The added classification head learns the mask classification task.

### Why resize images?

The classifier expects a consistent input shape of 160 by 160 RGB.

### Why use sigmoid?

There are two classes, so one sigmoid output can represent the
estimated probability of the positive class.

### Why binary cross-entropy?

It is a loss function for binary classification with a probability
output and binary labels.

### What is validation accuracy?

It is the fraction of correct predictions on the validation split.

### Why can webcam predictions differ from validation results?

Live scenes can have different lighting, pose, camera quality and
face crops compared with dataset images.

### Why select the lowest validation loss?

That was the configured checkpoint criterion. It evaluates how
well predicted probabilities match the validation labels.

### Does the deployment checker run inference?

No. It checks static files and model manifest structure.
Browser testing is needed to verify inference.

### Why is HTTPS needed for public camera use?

Browser camera access requires a secure context. HTTPS provides
that for public hosting; localhost supports local development.

### Did you implement helmet detection?

No. This version implements face mask classification.
A helmet classifier is proposed future work.

### What did you personally contribute?

I completed deployment documentation, macOS browser verification
and a static deployment checker, with merged GitHub PR evidence.