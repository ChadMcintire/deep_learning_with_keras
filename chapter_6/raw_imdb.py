import os
import numpy as np

############################
# Read in Data
##############################
imdb_dir = '../data/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

# print the average size of a document in words
print(np.mean([len(i) for i in texts]))


###########################
# Tokenize the text
###########################
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

#consider only the top 10000 words
tokenizer = Tokenizer(num_words=max_words)

#Updates internal vocabulary based on a list of texts.o
#In the case where texts contains lists, we assume each entry of the lists to be a token.
tokenizer.fit_on_texts(texts)

#Updates internal vocabulary based on a list of sequences.
#texts_to_sequences Transforms each text in texts to a sequence of integers. 
#So it basically takes each word in the text and replaces it with 
#its corresponding integer value from the word_index dictionary
#https://stackoverflow.com/questions/51956000/what-does-keras-tokenizer-method-exactly-do
sequences = tokenizer.texts_to_sequences(texts)

# It will create a dictionary s.t. word_index["the"] = 1; 
# every word gets a unique integer value.
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#This function transforms a list (of length num_samples) of sequences 
#(lists of integers) into a 2D Numpy array of shape (num_samples, num_timesteps). 
#num_timesteps is either the maxlen argument if provided, or the length 
#of the longest sequence in the list.
data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print("Shape of data tensor:", data.shape)
print("Shape of leabel tensor:", labels.shape)

#make an array from 0 to data_shape
indices = np.arange(data.shape[0])

#shuffle the data
np.random.shuffle(indices)

#change the indicies to the shuffled indices
data = data[indices]

#change the labels to the shuffled index
labels = labels[indices]

#make a training set from 0 to training_samples length-1
x_train = data[:training_samples]
y_train = labels[:training_samples]

#make a validation set from training_sample length to the validation
# + training length
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

####################################
#Map words to vector representation 
####################################

glove_dir = "../data/glove.6B"


#make an index that maps words to their vector representation

embeddings_index = {}
f = open(os.path.join(glove_dir, "glove.6B.100d.txt"))
for line in f:
    #split each line on whitespace
    values = line.split()
    #the word is contained in the first spot
    word = values[0]
    # the remaining values in the array are the coefficient 
    coefs = np.asarray(values[1:], dtype='float32')
    #we add to the dictionary using the word as the key and the 
    # the coefficient as the values
    embeddings_index[word] = coefs

f.close()

print('Found %s word vectors.' % len(embeddings_index))

###########################################
# Preparing the GloVe word-embedding matrix
###########################################

embedding_dim = 100

#max_words = 10000
embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in word_index.items():
    if i < max_words:
        #while i < 10000 add a word from the word dictionary
        # and 
        embedding_vector = embeddings_index.get(word)
        #excellent usecase for the walrus operator
        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

print(np.shape(embedding_matrix))

#########################################
# make the sequential 
########################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

########################################
#
########################################

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

########################################
#
########################################

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

model.save_weights('pre_trained_glove_model.h5')

#######################################
#
#######################################

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

########################################
#
########################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

###########################################
#
###########################################

test_dir = os.path.join(imdb_dir, 'test')

labels = []
texts = []


