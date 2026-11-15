import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, models, applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

import matplotlib.pyplot as plt

from leafthealthvision.data import load_datasets
from leafthealthvision.config.settings import EARLY_STOPPING_PATIENCE, IMAGE_RESCALE, IMAGE_TARGET_SIZE, OPTIMIZER, METRICS

train_ds, validation_ds, test_ds, class_names = load_datasets()

img_height, img_width = IMAGE_TARGET_SIZE

num_classes = len(class_names)

# --- Transfer learning: base pre-entrenado MobileNetV2 (guía TF) ---
base_model = applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet',
)

# Congelar la base: solo se entrena la cabeza nueva
base_model.trainable = False

inputs = Input(shape=(img_height, img_width, 3))
x = layers.Rescaling(IMAGE_RESCALE)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(num_classes)(x)
model = Model(inputs, outputs)

model.compile(optimizer=OPTIMIZER,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=METRICS)


model.summary()


epochs = 10
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
)
history = model.fit(
  train_ds,
  validation_data=validation_ds,
  epochs=epochs,
  callbacks=[early_stop]
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
