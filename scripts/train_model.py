"""
train_model.py — PlantCure CNN Training Script
================================================
Python 3.11+  |  TensorFlow 2.21  |  Keras 3.9+

Usage (run from project root):
    python scripts/train_model.py --dataset data/PlantVillage --epochs 10

Steps:
  1. Paste PlantVillage dataset into  data/PlantVillage/
  2. Run this script from the project root
  3. Model saved to  models/plant_model.h5
  4. class_labels.json auto-updated
  5. python app.py

Speed settings (defaults):
  --img-size 128   (was 224 — 3x fewer pixels per image)
  --batch    64    (was 32  — 2x fewer weight updates per epoch)
  --epochs   10    (was 30  — start fast, add more if accuracy low)
"""

import os, sys, argparse, json, random
import multiprocessing.pool
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# When script is in scripts/, BASE_DIR is the project root (one level up)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

parser = argparse.ArgumentParser(description="Train PlantCure CNN")
parser.add_argument("--dataset",   default=os.path.join(BASE_DIR, "data", "PlantVillage"))
parser.add_argument("--epochs",    type=int,   default=10)   # was 30
parser.add_argument("--batch",     type=int,   default=64)   # was 32
parser.add_argument("--img-size",  type=int,   default=128)  # was 224
parser.add_argument("--val-split", type=float, default=0.20)
parser.add_argument("--fine-tune", action="store_true")
parser.add_argument("--output",    default=os.path.join(BASE_DIR, "models", "plant_model.h5"))
parser.add_argument("--label-smoothing", type=float, default=0.08)
parser.add_argument("--resume", action="store_true",
                    help="Resume from last epoch checkpoint if available")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

try:
    import numpy as np
    try:
        import tensorflow as tf
    except Exception:
        tf = None
    try:
        import keras
        from keras import layers
        from keras.applications import MobileNetV2
        from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
        from keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
        # from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError as e:
    print(f"\nERROR: Missing dependency: {e}")
    print("Run: pip install tensorflow keras\n")
    sys.exit(1)

try:
    from keras.src.utils import dataset_utils as keras_dataset_utils
except Exception:
    try:
        from tensorflow.keras.utils import dataset_utils as keras_dataset_utils
    except Exception:
        keras_dataset_utils = None

IMG = (args.img_size, args.img_size)
os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
LOG_DIR = os.path.join(BASE_DIR, "training_logs")
os.makedirs(LOG_DIR, exist_ok=True)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
RESUME_MODEL_PATH = os.path.join(os.path.dirname(args.output) or ".", "plant_model_resume.keras")
STATE_PATH = os.path.join(LOG_DIR, "resume_state.json")

print("\n" + "=" * 58)
print("PlantCure - CNN Model Training")
print("=" * 58)
print(f"Dataset   : {args.dataset}")
print(f"Image size: {IMG}")
print(f"Epochs    : {args.epochs}")
print(f"Batch size: {args.batch}")
print(f"Fine-tune : {args.fine_tune}")
print(f"Output    : {args.output}")
print(f"LblSmooth : {args.label_smoothing}")
print(f"Resume    : {args.resume}")
print(f"Seed      : {args.seed}")
print("=" * 58 + "\n")

os.environ["PYTHONHASHSEED"] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
if tf is not None:
    try:
        tf.random.set_seed(args.seed)
    except Exception:
        pass


class _SequentialThreadPool:
    def __init__(self, *args, **kwargs):
        pass

    class _Result:
        def __init__(self, value):
            self._value = value

        def get(self, timeout=None):
            return self._value

    def map(self, func, iterable):
        return [func(item) for item in iterable]

    def imap(self, func, iterable):
        for item in iterable:
            yield func(item)

    def apply_async(self, func, args=(), kwds=None):
        kwds = kwds or {}
        return self._Result(func(*args, **kwds))

    def close(self):
        return None

    def join(self):
        return None

    def terminate(self):
        return None


multiprocessing.pool.ThreadPool = _SequentialThreadPool
if keras_dataset_utils is not None:
    keras_dataset_utils.ThreadPool = _SequentialThreadPool

if not os.path.isdir(args.dataset):
    print(f"❌  Dataset not found: {args.dataset}")
    print("    Paste PlantVillage into data/PlantVillage/ first.")
    sys.exit(1)

def resolve_dataset_root(path):
    # Handles nested folder exports like data/PlantVillage/PlantVillage
    # and Kaggle layouts like .../New Plant Diseases Dataset(Augmented)/train.
    root = os.path.abspath(path)
    for _ in range(6):
        dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        image_dirs = [
            d for d in dirs
            if len([f for f in os.listdir(os.path.join(root, d))
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]) > 0
        ]
        if image_dirs:
            return root

        lowered = {d.lower(): d for d in dirs}
        if "train" in lowered and os.path.isdir(os.path.join(root, lowered["train"])):
            root = os.path.join(root, lowered["train"])
            continue
        if len(dirs) == 1:
            root = os.path.join(root, dirs[0])
            continue
        break
    return root

dataset_root = resolve_dataset_root(args.dataset)

classes = sorted([d for d in os.listdir(dataset_root)
                  if os.path.isdir(os.path.join(dataset_root, d))
                  and not d.startswith(".")
                  and len([f for f in os.listdir(os.path.join(dataset_root, d))
                           if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]) > 0])
if not classes:
    print("❌  No class folders found inside dataset directory.")
    sys.exit(1)

print(f"OK: {len(classes)} classes found.")
print(f"Training from: {dataset_root}")
labels_path = os.path.join(os.path.dirname(args.output) or ".", "class_labels.json")
with open(labels_path, "w") as f:
    json.dump(classes, f, indent=2)
print(f"Labels saved -> {labels_path}\n")


def load_resume_state(path):
    if not os.path.isfile(path):
        return {"phase1_done": 0, "phase2_done": 0}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "phase1_done": int(data.get("phase1_done", 0)),
            "phase2_done": int(data.get("phase2_done", 0)),
        }
    except Exception:
        return {"phase1_done": 0, "phase2_done": 0}


def save_resume_state(path, phase1_done, phase2_done):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"phase1_done": int(phase1_done), "phase2_done": int(phase2_done)}, f, indent=2)


class ResumeStateCallback(keras.callbacks.Callback):
    def __init__(self, phase, state_path, p1_done, p2_done):
        super().__init__()
        self.phase = phase
        self.state_path = state_path
        self.p1_done = int(p1_done)
        self.p2_done = int(p2_done)

    def on_epoch_end(self, epoch, logs=None):
        # epoch is zero-based for the current fit call.
        done = int(epoch) + 1
        if self.phase == "phase1":
            save_resume_state(self.state_path, done, self.p2_done)
        else:
            save_resume_state(self.state_path, self.p1_done, done)


def find_backbone(model_obj):
    for lyr in model_obj.layers:
        if "mobilenet" in lyr.name.lower():
            return lyr
    raise RuntimeError("Could not locate MobileNet backbone in model.")

# ── Data generators ───────────────────────────────────────────────
train_ds = keras.utils.image_dataset_from_directory(
    dataset_root,
    labels="inferred",
    label_mode="categorical",
    class_names=classes,
    validation_split=args.val_split,
    subset="training",
    seed=args.seed,
    image_size=IMG,
    batch_size=args.batch,
)
val_ds = keras.utils.image_dataset_from_directory(
    dataset_root,
    labels="inferred",
    label_mode="categorical",
    class_names=classes,
    validation_split=args.val_split,
    subset="validation",
    seed=args.seed,
    image_size=IMG,
    batch_size=args.batch,
    shuffle=False,
)

rescale = layers.Rescaling(1 / 255)
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.12, 0.12),
    layers.RandomContrast(0.15),
], name="training_augment")

AUTOTUNE = tf.data.AUTOTUNE if tf is not None else None
train_ds = train_ds.map(lambda x, y: (rescale(augment(x, training=True)), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)
if AUTOTUNE is not None:
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

train_counts = {cls_name: 0 for cls_name in classes}
for _, labels in train_ds.unbatch():
    train_counts[classes[int(np.argmax(labels.numpy()))]] += 1
val_counts = {cls_name: 0 for cls_name in classes}
val_true = []
for _, labels in val_ds.unbatch():
    cls_idx = int(np.argmax(labels.numpy()))
    val_counts[classes[cls_idx]] += 1
    val_true.append(cls_idx)
val_true = np.array(val_true, dtype=int)

train_samples = sum(train_counts.values())
val_samples = sum(val_counts.values())
print(f"Training samples: {train_samples}  |  Validation samples: {val_samples}\n")

non_zero = [count for count in train_counts.values() if count > 0]
if non_zero:
    total = sum(non_zero)
    active = len(non_zero)
    class_weight = {
        idx: total / (active * train_counts[cls_name])
        for idx, cls_name in enumerate(classes)
        if train_counts[cls_name] > 0
    }
else:
    class_weight = None

print("Class distribution (training subset):")
for cls_name in classes:
    print(f"  - {cls_name}: {train_counts[cls_name]}")
print("")

resume_state = load_resume_state(STATE_PATH) if args.resume else {"phase1_done": 0, "phase2_done": 0}
p1_epochs = min(args.epochs, 10)  # cap phase-1 at 10 (was 20)
p2_epochs = max(0, args.epochs - p1_epochs)

# ── Build/load model ──────────────────────────────────────────────
if args.resume and os.path.isfile(RESUME_MODEL_PATH):
    print(f"Resuming model from: {RESUME_MODEL_PATH}")
    model = keras.models.load_model(RESUME_MODEL_PATH)
    base = find_backbone(model)
else:
    print("Building MobileNetV2 + custom head...\n")
    base = MobileNetV2(input_shape=IMG+(3,), include_top=False, weights="imagenet")
    base.trainable = False

    inputs  = keras.Input(shape=IMG+(3,))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(128, activation="relu")(x)
    x       = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(classes), activation="softmax")(x)
    model   = keras.Model(inputs, outputs)

model.summary()

# ── Phase 1 ───────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
    metrics=["accuracy"])

cbs1 = [
    ModelCheckpoint(args.output, save_best_only=True, monitor="val_accuracy", verbose=1),
    ModelCheckpoint(RESUME_MODEL_PATH, save_best_only=False, verbose=0),
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=4, min_lr=1e-6, verbose=1),
    CSVLogger(f"{LOG_DIR}/{run_id}_phase1.csv", append=args.resume),
    ResumeStateCallback("phase1", STATE_PATH, resume_state.get("phase1_done", 0), resume_state.get("phase2_done", 0)),
]
phase1_done = min(max(int(resume_state.get("phase1_done", 0)), 0), p1_epochs) if args.resume else 0
if phase1_done < p1_epochs:
    print(f"\nPhase 1: training head only ({phase1_done}/{p1_epochs} complete)...\n")
    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=phase1_done,
        epochs=p1_epochs,
        callbacks=cbs1,
        class_weight=class_weight
    )
else:
    print(f"\nPhase 1 already complete ({p1_epochs}/{p1_epochs}). Skipping.")

# ── Phase 2 (fine-tune) ───────────────────────────────────────────
if args.fine_tune and p2_epochs > 0:
    print("\nPhase 2: fine-tuning top 40 layers...\n")
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=["accuracy"])
    cbs2 = [
        ModelCheckpoint(args.output, save_best_only=True, monitor="val_accuracy", verbose=1),
        ModelCheckpoint(RESUME_MODEL_PATH, save_best_only=False, verbose=0),
        EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-8, verbose=1),
        CSVLogger(f"{LOG_DIR}/{run_id}_phase2.csv", append=args.resume),
        ResumeStateCallback("phase2", STATE_PATH, p1_epochs, resume_state.get("phase2_done", 0)),
    ]
    phase2_done = min(max(int(resume_state.get("phase2_done", 0)), 0), p2_epochs) if args.resume else 0
    if phase2_done < p2_epochs:
        model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=phase2_done,
            epochs=p2_epochs,
            callbacks=cbs2,
            class_weight=class_weight
        )
    else:
        print(f"Phase 2 already complete ({p2_epochs}/{p2_epochs}). Skipping.")

# ── Final evaluation ──────────────────────────────────────────────
print("\nFinal evaluation...")
model = keras.models.load_model(args.output)
loss, acc = model.evaluate(val_ds, verbose=0)
probs = model.predict(val_ds, verbose=0)
y_true = val_true
y_pred = np.argmax(probs, axis=1)
top3 = np.argsort(probs, axis=1)[:, -3:]
top3_hit = float(np.mean([int(y_true[i] in top3[i]) for i in range(len(y_true))])) if len(y_true) else 0.0

per_class = {}
for i, cls in enumerate(classes):
    mask = (y_true == i)
    total = int(mask.sum())
    if total == 0:
        per_class[cls] = {"support": 0, "accuracy": None}
        continue
    cls_acc = float((y_pred[mask] == i).sum() / total)
    per_class[cls] = {"support": total, "accuracy": round(cls_acc, 4)}

eval_report = {
    "run_id": run_id,
    "val_loss": round(float(loss), 6),
    "val_accuracy": round(float(acc), 6),
    "val_top3_accuracy": round(float(top3_hit), 6),
    "num_classes": len(classes),
    "per_class": per_class,
}
eval_path = os.path.join(LOG_DIR, f"{run_id}_eval.json")
with open(eval_path, "w", encoding="utf-8") as f:
    json.dump(eval_report, f, indent=2)

print("\n" + "=" * 58)
print("Training complete")
print("=" * 58)
print(f"Validation accuracy : {acc*100:.2f}%")
print(f"Validation top-3 acc: {top3_hit*100:.2f}%")
print(f"Model saved to      : {args.output}")
print(f"Labels file         : {labels_path}")
print(f"Eval report         : {eval_path}")
print("\nNext steps:")
print("1. Run: python app.py")
print("2. Open http://localhost:5000")
print("=" * 58)
