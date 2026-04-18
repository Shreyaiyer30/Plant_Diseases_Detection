"""
train_optimized.py — PlantCure Fast CNN (Simple Architecture)
==============================================================
Use this instead of train_model.py for a much faster training run on CPU.

Differences vs train_model.py:
  - No MobileNetV2 transfer learning (avoids large pretrained weights download)
  - Simple 3-block CNN built from scratch — lighter, faster
  - Image size: 128x128  (vs 224x224)
  - Batch size: 64        (vs 32)
  - Epochs:     10        (start here, increase if accuracy is low)
  - Uses ImageDataGenerator (compatible with older TF/Keras versions)
  - Saves to models/plant_model.h5  (same path app.py expects)

Accuracy expectation:
  - Simple CNN @ 10 epochs: ~70-80% validation accuracy
  - For higher accuracy: use train_model.py with MobileNetV2 (transfer learning)

Run from project root:
    python scripts/train_optimized.py

Optional arguments:
    python scripts/train_optimized.py --epochs 15 --dataset data/PlantVillage
"""

import os, sys, json
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ── Resolve project root (this file is in scripts/, root is one level up) ──
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
LOG_DIR     = os.path.join(BASE_DIR, "training_logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

# ── CLI args (minimal) ──────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser(description="PlantCure Fast CNN Trainer")
parser.add_argument("--dataset",  default=os.path.join(BASE_DIR, "data", "PlantVillage"))
parser.add_argument("--epochs",   type=int,   default=10)
parser.add_argument("--batch",    type=int,   default=64)
parser.add_argument("--img-size", type=int,   default=128)
parser.add_argument("--output",   default=os.path.join(MODELS_DIR, "plant_model.h5"))
args = parser.parse_args()

IMG_SIZE   = args.img_size
BATCH_SIZE = args.batch

# ── Import TF/Keras ─────────────────────────────────────────────────────────
try:
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    print(f"TensorFlow {tf.__version__} | Keras {keras.__version__}")
except ImportError as e:
    print(f"\nERROR: {e}")
    print("Run: pip install tensorflow\n")
    sys.exit(1)

# ── CPU Threading Optimizations ─────────────────────────────────────────────
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(4)
print("CPU threading: intra=8, inter=4")

# ── Resolve dataset root (handles nested Kaggle layouts) ────────────────────
def resolve_dataset_root(path):
    root = os.path.abspath(path)
    for _ in range(6):
        dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        image_dirs = [
            d for d in dirs
            if any(f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                   for f in os.listdir(os.path.join(root, d)))
        ]
        if image_dirs:
            return root
        lowered = {d.lower(): d for d in dirs}
        if "train" in lowered:
            root = os.path.join(root, lowered["train"])
            continue
        if len(dirs) == 1:
            root = os.path.join(root, dirs[0])
            continue
        break
    return root

if not os.path.isdir(args.dataset):
    print(f"\n❌  Dataset not found: {args.dataset}")
    print("    Place PlantVillage into data/PlantVillage/ first.\n")
    sys.exit(1)

dataset_root = resolve_dataset_root(args.dataset)
print(f"Dataset root resolved: {dataset_root}")

# Discover and sort class folders
classes = sorted([
    d for d in os.listdir(dataset_root)
    if os.path.isdir(os.path.join(dataset_root, d))
    and not d.startswith(".")
    and any(f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            for f in os.listdir(os.path.join(dataset_root, d)))
])
if not classes:
    print("❌  No class folders found with images.")
    sys.exit(1)

print(f"\n{'='*58}")
print(f"PlantCure — Fast CNN Trainer (Simple Architecture)")
print(f"{'='*58}")
print(f"Dataset   : {dataset_root}")
print(f"Classes   : {len(classes)}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch     : {BATCH_SIZE}")
print(f"Epochs    : {args.epochs}")
print(f"Output    : {args.output}")
print(f"{'='*58}\n")

# Save class labels (so app.py can read them)
labels_path = os.path.join(MODELS_DIR, "class_labels.json")
with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(classes, f, indent=2)
print(f"✓ Labels saved → {labels_path}  ({len(classes)} classes)\n")

# ── Data generators with augmentation ───────────────────────────────────────
datagen_train = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.20,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)
datagen_val = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.20,
)

train_gen = datagen_train.flow_from_directory(
    dataset_root,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=classes,
    subset="training",
    shuffle=True,
    seed=42,
)
val_gen = datagen_val.flow_from_directory(
    dataset_root,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=classes,
    subset="validation",
    shuffle=False,
    seed=42,
)

print(f"Training samples  : {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}\n")

# ── Build model — simple 3-block CNN ────────────────────────────────────────
model = keras.Sequential([
    # Block 1
    keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                        input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),

    # Block 2
    keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),

    # Block 3
    keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),

    # Block 4 (extra depth to handle 38 classes)
    keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.GlobalAveragePooling2D(),   # much fewer params than Flatten

    # Head
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(classes), activation="softmax"),
], name="PlantCure_SimpleCNN")

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ── Callbacks ────────────────────────────────────────────────────────────────
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
callbacks = [
    keras.callbacks.ModelCheckpoint(
        args.output,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
    keras.callbacks.CSVLogger(
        os.path.join(LOG_DIR, f"{run_id}_optimized.csv")
    ),
]

# ── Train ────────────────────────────────────────────────────────────────────
print(f"\nStarting training ({args.epochs} epochs)...\n")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=args.epochs,
    callbacks=callbacks,
    verbose=1,
)

# ── Final report ─────────────────────────────────────────────────────────────
best_val_acc = max(history.history.get("val_accuracy", [0])) * 100

print(f"\n{'='*58}")
print("✅  Training complete!")
print(f"{'='*58}")
print(f"Best validation accuracy : {best_val_acc:.2f}%")
print(f"Model saved → {args.output}")
print(f"Labels saved → {labels_path}")
print(f"\nNext steps:")
print(f"  1. Restart the app:  python app.py")
print(f"  2. Open:             http://localhost:5000")
if best_val_acc < 75:
    print(f"\n  ℹ️  Accuracy ({best_val_acc:.1f}%) is decent but you can improve it:")
    print(f"     - Run more epochs:  python scripts/train_optimized.py --epochs 20")
    print(f"     - Or use MobileNetV2 transfer learning:")
    print(f"       python scripts/train_model.py --dataset data/PlantVillage --epochs 10")
print(f"{'='*58}")
