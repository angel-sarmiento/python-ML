#%%
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
#%%
#importing the fashion dataset
fashion_mnist = keras.datasets.fashion_mnist

#loading the dataset, returning four numpy arrays
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#class names are not included with the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#some code to inspect the first image of the training set
#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

#each image is scaled from 0 to 255 in terms of pixels. So we divide to make them 0 to 1.
#TRAINING AND TEST SETS NEED SAME PREPROCESSING
train_images = train_images / 255.0
test_images = test_images / 255.0
#%%
#display 25 images to show we have the correct format
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#%%
#building the model
model = keras.Sequential([
    #first layer "flattens" the 2-d Matrix of images into 784 different pixels (1-D)
    keras.layers.Flatten(input_shape=(28,28)),
    #this is the first layer where processing occurs, 128 neurons, and ReLU applied
    keras.layers.Dense(128, activation=tf.nn.relu),
    #second and final layer has 10 neurons (0 - 9) with softmax activation. softmax turns it into a probability distribution
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
#This is where we establish the loss function, optimizer (what the model does in response to loss function, think gradient descent), 
#and the metric, here being accuracy.8
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#using the model.fit method, we fit this model to the training data.
model.fit(train_images, train_labels, epochs=5)

#%%
#now to test the model on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test Accuracy:', test_acc)

#making predictions
predictions = model.predict(test_images)

#number 0-9 with highest probability and highest confidence value, for the first image, a boot, it should return 9 for the category ankle boot.
np.argmax(predictions[0])

#%%
#here are some functions to plot the image alongside a value array to show the probabilities of each type for an image
##KEEP THESE FUNCTIONS FOR FUTURE REFERENCE. PRETTY USEFUL VISUALISATIONS CAN BE CREATED FROM THESE
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#%%
#now we can plot any number of the test images along with the predicted label and the true label. Correct in blue, 
#incorrect in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i,predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()

#%% Dropout
##LETS ADDRESS OVERFITTING WITH DROPOUT
dpt_model = keras.models.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(input_shape=(28,28)),

        keras.layers.Dense(10, activation=tf.nn.softmax)      
])
dpt_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

dpt_model_history = dpt_model.fit(train_images, train_labels, epochs=5)

#%%Results
test_loss, test_acc = dpt_model.evaluate(test_images, test_labels)

print('Test Accuracy:', test_acc)