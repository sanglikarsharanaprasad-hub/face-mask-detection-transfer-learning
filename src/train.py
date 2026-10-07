import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
tf.keras.utils.set_random_seed(42)

# Identical seed gives complementary training and validation splits.
settings = dict(
    directory=str(DATA),
    class_names=["with_mask", "without_mask"],
    labels="inferred",
    label_mode="binary",
    validation_split=0.2,
    seed=42,
    image_size=(160, 160),
    batch_size=32,
)

train = tf.keras.utils.image_dataset_from_directory(
    subset="training", **settings
)
validation = tf.keras.utils.image_dataset_from_directory(
    subset="validation", **settings
)

train = train.prefetch(tf.data.AUTOTUNE)
validation = validation.prefetch(tf.data.AUTOTUNE)

# Download ImageNet weights on the first run.
base = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

inputs = layers.Input(shape=(160, 160, 3))
# Preprocessing is saved inside the model for consistent webcam input.
x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train,
    validation_data=validation,
    epochs=5,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS / "mask_detector.keras"),
            monitor="val_loss",
            save_best_only=True,
        )
    ],
)

(MODELS / "class_names.json").write_text(
    json.dumps(["with_mask", "without_mask"], indent=2),
    encoding="utf-8",
)
(RESULTS / "history.json").write_text(
    json.dumps(history.history, indent=2),
    encoding="utf-8",
)

epochs = range(1, len(history.history["accuracy"]) + 1)

for metric, title in [
    ("accuracy", "Training and Validation Accuracy"),
    ("loss", "Training and Validation Loss"),
]:
    plt.figure()
    plt.plot(epochs, history.history[metric], label="Training")
    plt.plot(epochs, history.history[f"val_{metric}"], label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.xticks(list(epochs))
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / f"{metric}.png")
    plt.close()

best_model = tf.keras.models.load_model(MODELS / "mask_detector.keras")
loss, accuracy = best_model.evaluate(validation)
(RESULTS / "validation_metrics.json").write_text(
    json.dumps({"loss": float(loss), "accuracy": float(accuracy)}, indent=2),
    encoding="utf-8",
)

print(f"\nBest model validation accuracy: {accuracy:.2%}")
print("Model saved: models/mask_detector.keras")
print("Plots saved: results/accuracy.png and results/loss.png")