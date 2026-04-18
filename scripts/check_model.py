import os, json, sys

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

model_path  = "models/plant_model"
labels_path = "models/class_labels.json"

print("=== MODEL FILE CHECK ===")
exists = os.path.exists(model_path)
print(f"Exists : {exists}")
if exists:
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"Size   : {size_mb:.2f} MB")

print()
print("=== LABELS FILE CHECK ===")
labels = []
if os.path.exists(labels_path):
    with open(labels_path) as f:
        labels = json.load(f)
    print(f"Labels count : {len(labels)}")
    for i, l in enumerate(labels):
        print(f"  [{i:02d}] {l}")
else:
    print("labels file NOT found")

print()
print("=== LOADING MODEL ===")
try:
    import tensorflow as tf
    print(f"TF version: {tf.__version__}")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Load: OK")
    print(f"Input shape    : {model.input_shape}")
    print(f"Output shape   : {model.output_shape}")
    output_classes = model.output_shape[-1]
    print(f"Output neurons : {output_classes}")
    print(f"Labels in JSON : {len(labels)}")
    print(f"Match          : {output_classes == len(labels)}")
    if output_classes != len(labels):
        print()
        print("!!! MISMATCH: model has", output_classes, "outputs but labels has", len(labels), "entries")
        print("    You must retrain with the full dataset OR fix class_labels.json")
except Exception as e:
    print(f"LOAD ERROR: {e}")
