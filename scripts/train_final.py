import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import json

# --- USER UPDATED PARAMETERS (ASCII version for Windows stability) ---
DATA_DIR = 'data/PlantVillage'
MODELS_DIR = 'models'
MODEL_NAME = 'combined_plant_model_final.h5'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
MAX_IMAGES_PER_CLASS = 100

def train_custom_model():
    print("="*60)
    print("STARTING CUSTOM TRAINING (Master Model)")
    print(f"Params: 100 images/class | 128x128 | 10 Epochs")
    print("="*60)
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    print(f"Scanning folders in {DATA_DIR}...")
    
    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    num_classes = len(train_generator.class_indices)
    total_target_images = num_classes * MAX_IMAGES_PER_CLASS
    steps_per_epoch = int((total_target_images * 0.8) // BATCH_SIZE)
    
    val_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    classes = list(train_generator.class_indices.keys())
    labels_path = os.path.join(MODELS_DIR, 'class_labels_final.json')
    with open(labels_path, 'w') as f:
        json.dump(classes, f)
    print(f"Class labels saved. Found {len(classes)} classes.")

    print("Building MobileNetV2 architecture...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, 
                             input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', metrics=['accuracy'])

    print("Training starting... (Estimated: 45-60 mins on CPU)")
    
    model.fit(
        train_generator,
        steps_per_epoch=max(1, steps_per_epoch),
        validation_data=val_generator,
        validation_steps=max(1, int(steps_per_epoch * 0.2)),
        epochs=EPOCHS
    )

    save_path = os.path.join(MODELS_DIR, MODEL_NAME)
    model.save(save_path)
    print(f"SUCCESS! Model saved to {save_path}")

if __name__ == "__main__":
    train_custom_model()
