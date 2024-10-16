import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization 
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import RandomFlip, RandomRotation
from tensorflow.keras.regularizers import L2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_cats = len(os.listdir("./datasets/training_set/cats"))
train_dogs = len(os.listdir("./datasets/training_set/dogs"))

print(f"The training dataset contain {train_cats} cat images and {train_dogs} dog images")

test_cats = len(os.listdir("./datasets/test_set/cats"))
test_dogs = len(os.listdir("./datasets/test_set/dogs"))

print(f"The test dataset contain {test_cats} cat images and {test_dogs} dog images")

IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
  directory = "./datasets/training_set",
  color_mode = "rgb",
  batch_size =BATCH_SIZE,
  image_size =(IMG_HEIGHT, IMG_WIDTH),
  shuffle = True,
  seed = 123,
  validation_split = 0.2,
  subset = 'training',
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
  directory = "./datasets/training_set",
  color_mode = "rgb",
  batch_size =BATCH_SIZE,
  image_size =(IMG_HEIGHT, IMG_WIDTH),
  shuffle = True,
  seed = 123,
  validation_split = 0.2,
  subset = 'training',
)

test_ds = tf.keras.utils.image_dataset_from_directory(
  directory = "./datasets/test_set",
  color_mode = 'rgb',
  batch_size = BATCH_SIZE,
  image_size = (IMG_HEIGHT, IMG_WIDTH),
  shuffle = True,
)

num_classes = len(train_ds.class_names)
class_names = train_ds.class_names
print(class_names)

model = Sequential([
  Rescaling(1./255),
  Conv2D(16,3, padding = 'same', activation = 'relu'),
  MaxPooling2D(),
  Conv2D(32,3, padding = 'same', activation = 'relu'),
  MaxPooling2D(),
  Conv2D(64,3, padding = 'same', activation = 'relu'),
  MaxPooling2D(),
  Flatten(),
  Dropout(0.2),
  Dense(128),
  Dense(num_classes)
])
model.compile(optimizer = 'adam',
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
  metrics = ['accuracy'])

model.summary()

EPOCHS = 10

history = model.fit(
  train_ds,
  validation_data = valid_ds,
  epochs = EPOCHS
)

epochs_range = range(EPOCHS)

plt.figure(figsize=(8,8))

plt.subplot(1,2,1)
plt.plot(epochs_range, history.history['accuracy'], label = 'Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label = 'Validation Accuracy')
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, history.history['loss'], label = 'Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label = 'Validation Loss')
plt.title('Loss')
plt.show()

model.evaluate(test_ds, batch_size = 32, verbose=1)

model.save('dog_cat_model.keras')
