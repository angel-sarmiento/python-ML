#%%
import tensorflow as tf
from tensorflow import keras
tf.enable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
print(tf.__version__)

#this program is the usage of tensorflow in the interest of text classification;
#the objective being to identify if words or sequences of words are negative or positive.
#i Will be using the IMDB dataset that is packaged with Tensorflow
#%%
#downloading the dataset
imdb = keras.datasets.imdb
#loading the dataset with labels; num_words keeps the most frequently occurring words. Each word in the dataset is an integer representing 
#a word in a dictionary
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
#%%
##Here is a way to convert the integers of the dataset back into text
#dictionary mapping words to an integer index
word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])
#this is the function that can be applied to see all of the words
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
#testing the first review in the dataset
decode_review(train_data[0])

#in order to use a neural network, the movie reviews must all be the same length. pad_sequences standardizes the lengths
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)
#the same preprocessing must be done to the training and test sets
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)
#%%
##BUILDING THE MODEL
vocab_size = 10000

model = keras.Sequential()
#embedding layer. dimensions are (batch, sequence, embedding).  This layer takes the integer-encoded vocabulary and looks up the embedding vector for each word-index
model.add(keras.layers.Embedding(vocab_size, 16))
#Global Average Pooling 1D layer returns a fixed-length vector for each item by averaging over the sequence.
model.add(keras.layers.GlobalAveragePooling1D())
#Dense layer with 16 hidden units
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
#Sigmoid activation function layer. Outputs a float between 0 and 1. You know this. Binary classifiers benefit from using sigmoid. Softmax for multiclass with Cross Entropy cost
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

#now to configure an optimizer and loss function
#Adam is a well-known and well-performing alternative to stochastic gradient descent that is mor efficient. loss function is binary crossentropy because it is very usefu
#for probabilities. Metric is accuracy
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])

#a validation set will be created to evaluate the model on data it has not seen before. 10000 other examples from the original data
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

#training the model for 40 epochs using mini batches
history = model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

#%% evaluating the model
results = model.evaluate(test_data, test_labels)
print(results)
#%%

#creating some graphs for accuracy
history_dict = history.history
history_dict.keys()
#creating the basic plot
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label= 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
   
#adding the data into the plot
plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#%%
#going to retrain a second model with L2 Regularizes to address the overfitting

l2_model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size, 16),
        #Global Average Pooling 1D layer returns a fixed-length vector for each item by averaging over the sequence.
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu),
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

l2_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'binary_crossentropy'])


#same paramaters as the model fit above
l2_model_history = l2_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(x_val, y_val), verbose=2)

#%% Test data evaluation
resultsl2 = l2_model.evaluate(test_data, test_labels)
print()
