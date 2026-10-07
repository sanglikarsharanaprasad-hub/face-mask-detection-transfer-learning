# MaskLab browser demo

Uses the project’s trained 160×160 MobileNetV2 mask classifier, converted to TensorFlow.js. BlazeFace detects face crops. All models and scripts are served locally; images stay on your device.

Run from the repository root:

```powershell
python -m http.server 8000 --directory web
```

Open http://localhost:8000 in Chrome or Edge. Wait for Model ready, start the camera and allow access. Stop camera releases it. Image mode and snapshot downloads are also available.

The 98.94% figure is image validation accuracy, not a live webcam accuracy measurement.

Hosted demo: https://masklab-live.vishnuyadav-venkates.chatgpt.site
