#%% Libraries
import os

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt 

#%% importing the data
base_dir = r'C:\Users\Angel\Documents\Python Scripts\Tensorflow\dogs-vs-cats'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat/dog pictures
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat/dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

#%% See the names of files in the directories
train_cat_fnames = os.listdir( train_cats_dir )
train_dog_fnames = os.listdir( train_dogs_dir )

print(train_cat_fnames[:10])
print(train_dog_fnames[:10])



#%%Data preprocessing
#normalizing the image data
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#%% Building the training and validation generators 
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

#%% Building a model
model = keras.models.Sequential([
    #input shape is 150 x 150 with 3 bytes color
    keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(32, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    #flatten
    keras.layers.Flatten(),
    #512 neuron hidden layer
    keras.layers.Dense(512, activation='relu'),
    #sigmoid final layer. One neuron, 0-1 for 'cats' or 'dogs' respectively
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

#%%compiling the model
from tensorflow.keras.optimizers import RMSprop

opt = RMSprop(lr=0.0001)
model.compile(optimizer= opt, loss='binary_crossentropy', metrics=['acc'])

#%%Training
history = model.fit_generator(train_generator, validation_data=validation_generator, steps_per_epoch=100, epochs=15, validation_steps=50, verbose=2)

#%% Prediction
