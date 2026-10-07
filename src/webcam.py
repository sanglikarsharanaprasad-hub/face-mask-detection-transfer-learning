import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
model = tf.keras.models.load_model(
    ROOT / "models" / "mask_detector.keras",
    compile=False,
)

# OpenCV includes this face detector.
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if detector.empty():
    raise RuntimeError("Could not load the face detector.")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Cannot open webcam. Close other camera apps.")

print("Press Q to quit. Press S to save a demo screenshot.")

try:
    while True:
        success, frame = camera.read()
        if not success:
            print("Could not read a camera frame.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        for x, y, w, h in faces:
            face = frame[y:y + h, x:x + w]
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(
                rgb, (160, 160), interpolation=cv2.INTER_LINEAR
            )
            batch = np.expand_dims(resized.astype("float32"), axis=0)

            # Model already contains the required pixel normalization.
            probability = float(model(batch, training=False).numpy()[0, 0])
            no_mask = probability >= 0.5
            label = "No Mask" if no_mask else "Mask"
            confidence = probability if no_mask else 1 - probability
            color = (0, 0, 255) if no_mask else (0, 200, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{label}: {confidence:.1%}",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        if len(faces) == 0:
            cv2.putText(
                frame, "No face detected", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2
            )

        cv2.putText(
            frame, "Q: Quit | S: Screenshot", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        cv2.imshow("Face Mask Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("s"), ord("S")):
            folder = ROOT / "screenshots"
            folder.mkdir(exist_ok=True)
            filename = folder / (
                datetime.now().strftime("demo_%Y%m%d_%H%M%S_%f") + ".png"
            )
            if cv2.imwrite(str(filename), frame):
                print(f"Screenshot saved: {filename}")

finally:
    camera.release()
    cv2.destroyAllWindows()