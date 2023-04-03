##########################################
#   ___ __  __ ___  ___  ___ _____ ___   #
#  |_ _|  \/  | _ \/ _ \| _ |_   _/ __|  #
#   | || |\/| |  _| (_) |   / | | \__ \  #
#  |___|_|  |_|_|  \___/|_|_\ |_| |___/  #
#                                        #
##########################################

import os
import random
import datetime

import numpy as np
import tensorflow as tf

from tensorflow import keras as krs
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

import config

#############################################
#   ___ _____ _   _  _ ___   _   ___ ___    #
#  / __|_   _/_\ | \| |   \ /_\ | _ |   \   #
#  \__ \ | |/ _ \| .` | |) / _ \|   | |) |  #
#  |___/ |_/_/ \_|_|\_|___/_/ \_|_|_|___/   #
#                                           #
#############################################

mnist = tf.keras.datasets.mnist

(training_images, training_labels), \
    (test_images, test_labels) = mnist.load_data()

training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0

test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0
     
def create_model(summary=False, name=None):
  model = tf.keras.models.Sequential([
    krs.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    krs.layers.MaxPooling2D(2, 2),
    krs.layers.Flatten(),
    krs.layers.Dense(128, activation='relu'),
    krs.layers.Dense(64, activation='relu'),
    krs.layers.Dense(10, activation='softmax')
  ], name=name)
  if summary==True:
    model.summary()
  return model

model = create_model(summary=True)


log_dir="logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
chkpoint = './base.model'
train_csv = "training_csv.log"

checkpoint = ModelCheckpoint(
    chkpoint,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    period=1
)

csvlogger = CSVLogger(
    filename= train_csv,
    separator = ",",
    append = False
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


callbacks = [tensorboard_callback]




# Compiling the Model
model.compile(optimizer=krs.optimizers.Adam(lr=lr),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

training = model.fit(training_images, 
                     training_labels, 
                     epochs=n_epochs,
                     #steps_per_epoch=steps_epoch,
                     verbose=verbosity,
                     callbacks=callbacks)

## Saving and Loading the WEIGHTS
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')


# Saving and Loading the whole model (ARCHITECTURE + WEIGHTS + OPTIMIZER STATE)
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

print("SAMPLE")