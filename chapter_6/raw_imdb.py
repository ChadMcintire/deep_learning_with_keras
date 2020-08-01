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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence importpad_sequences
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
sequences = tokenizer.texts_to_sequences(text)

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



