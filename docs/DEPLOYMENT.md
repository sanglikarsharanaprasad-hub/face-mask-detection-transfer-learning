# MaskLab Deployment Guide

## Overview

MaskLab detects faces and classifies them as Mask or No Mask.

The website runs in the browser using TensorFlow.js. BlazeFace
locates faces, and the converted MobileNetV2 classifier predicts
whether each detected face is wearing a mask.

Website inference does not require a Python or Flask server.

## Website files

The entire website is contained in the `web/` directory:

- `index.html`: page structure.
- `style.css`: interface styling.
- `app.js`: camera controls and prediction logic.
- `model/`: converted mask classifier and its weight files.
- `face-model/`: face detector and its weight file.
- `vendor/`: locally bundled JavaScript libraries.

Keep these directories together when deploying. Missing model
files or weight files will prevent detection from working.

## Run locally on macOS

From the repository root, run:

```bash
python3 -m http.server 8000 --directory web
```

Open:

http://localhost:8000

Keep the terminal running while using the website.
Press Control + C in the terminal to stop the server.

Do not open `index.html` directly using a `file://` URL.
Serve the website through localhost or an HTTPS host.

## Camera permissions

1. Open the website.
2. Start the camera using the interface.
3. Allow the browser to access the camera.
4. Wait for the models to finish loading.
5. Position your face clearly in the camera view.

For public deployment, use HTTPS so the browser can request
camera access. Local development can use localhost.

On macOS, check System Settings > Privacy & Security > Camera
if camera access is blocked for your browser.

## Deployment requirements

Deploy the contents of `web/` as a static website.

The host must serve:

- HTML, CSS and JavaScript files.
- Both model JSON files.
- Every corresponding `.bin` weight file.
- The bundled libraries in `vendor/`.

Preserve the directory structure and filename capitalization.
The deployed page must load over HTTPS.

## Vercel status

Vercel deployment has not yet been verified as part of this guide.
After deployment, record the public URL and complete the checks below.

## Manual verification checklist

Record the actual result of each check after testing:

- The page loads without missing assets.
- Both models load successfully.
- Camera permission can be granted.
- A visible face receives a prediction.
- Mask and No Mask examples can be demonstrated.
- Stopping the camera releases camera access.
- Image upload produces a prediction when a face is detected.

A working camera preview alone does not confirm model inference.

## Troubleshooting

### Camera does not start

Check browser and operating-system camera permissions.
Close other applications using the camera.
Use HTTPS or localhost.

### Camera works but no prediction appears

Wait for model loading to finish.
Make sure the face is visible and well lit.
Check the browser developer console for errors.

### Model loading fails

In the browser developer tools, check the Network tab for failed
requests. Confirm that model JSON files, weight files and vendor
scripts are present at their expected relative paths.

### Website shows a directory listing or a missing page

Confirm that the deployed website directory contains `index.html`.
For local testing, run the server command from the repository root.

## Limitations

The saved model achieved approximately 98.94% validation accuracy.
This is a validation-set result, not a guarantee of webcam accuracy.

Lighting, face angle, distance and face detection quality can affect
predictions. This version classifies face masks; helmet detection
is not implemented.
## Local browser verification

Tested by Vishnu Yadav V on macOS on 7 October 2026 using
http://localhost:8000.

- Camera access and preview worked.
- Predictions appeared for mask and no-mask examples.
- The Stop Camera control worked.

These were manual functional checks, not an accuracy benchmark.
Vercel deployment remains unverified.
