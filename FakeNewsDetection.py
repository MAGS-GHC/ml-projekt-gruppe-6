# Python 3.9.13
# Importing relevant libraries
import pandas as pd
from sklearn import preprocessing 
import tensorflow as tf
import numpy as np 



#Reading data from csv file
data = pd.read_csv("news.csv")

#Printing head and find out 1 unnamed column
#print(data.head())

#Dropping(deleting) unnamed column
data = data.drop(["Unnamed: 0"], axis=1)
#print(data.head(5))

"""
Data enconding
"""
le = preprocessing.LabelEncoder()
le.fit(data['label'])
data['label'] = le.transform(data['label'])


"""
variables required for the model training
"""
embedding_dim = 50
max_length = 54
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 3000
test_portion = .1


"""
Tokenization

Divides a large piece of continuous text into distinct units or tokens
"""
title = []
text = []
labels = []
for x in range(training_size):
    title.append(data['title'][x])
    text.append(data['text'][x])
    labels.append(data['label'][x])


"""
Applying Tokenization
"""
tokenizer1 = tf.keras.preprocessing.text.Tokenizer()
tokenizer1.fit_on_texts(title)
word_index1 = tokenizer1.word_index
vocab_size1 = len(word_index1)
sequences1 = tokenizer1.texts_to_sequences(title)
padded1 = tf.keras.preprocessing.sequence.pad_sequences(sequences1, padding=padding_type, truncating=trunc_type)
split = int(test_portion * training_size)
traning_sequences1 = padded1[split:training_size]
test_sequences1 = padded1[0:split]
test_labels = labels[0:split]
training_labels = labels[split:training_size]


"""
Generating Word Embedding

It allows words with similar meanings to have a similar representation
we will use glove.6B.50d.txt. It has the predefined vector space for words
"""

embedding_index = {}
with open('glove.6B.50d.txt', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs


#Generating embeddings
        
embeddings_matrix = np.zeros((vocab_size1 + 1, embedding_dim))
for word, i in word_index1.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector

"""
Creating Model Architecture

We map original input data into some set of real-valued dimensions
"""



model = tf.keras.Sequential([ 
    tf.keras.layers.Embedding(vocab_size1+1, embedding_dim, 
                              input_length=max_length, weights=[ 
                                  embeddings_matrix], 
                              trainable=False), 
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Conv1D(64, 5, activation='relu'), 
    tf.keras.layers.MaxPooling1D(pool_size=4), 
    tf.keras.layers.LSTM(64), 
    tf.keras.layers.Dense(1, activation='sigmoid') 
]) 
model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['accuracy']) 
model.summary() 



#print("Hello, World!")