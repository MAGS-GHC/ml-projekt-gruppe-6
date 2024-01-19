from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 

"""
Tokenization 
"""
def tokenize_and_sequence(data, padding_type, trunc_type, test_portion, training_size):
	title = [] 
	text = [] 
	labels = []
	for x in range(training_size): 
		title.append(data['title'][x]) 
		text.append(data['text'][x]) 
		labels.append(data['label'][x])
    
    #Applying Tokenization
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(title)
	word_index = tokenizer.word_index
	vocab_size = len(word_index)
	sequences = tokenizer.texts_to_sequences(title)
	padded = pad_sequences(
        sequences, padding=padding_type, truncating=trunc_type)
	split = int(test_portion * training_size)
	training_sequences = padded[split:training_size]
	test_sequences = padded[0:split]
	test_labels = labels[0:split]
	training_labels = labels[split:training_size]
	
	return tokenizer, word_index, vocab_size, training_sequences, test_sequences, test_labels, training_labels


"""
Generating Word Embedding
"""
def generate_word_embeddings(word_index, embedding_dim):
	embeddings_index = {}
	with open('files/glove.6B.50d.txt') as f:
	        for line in f:
	            values = line.split() 
	            word = values[0] 
	            coefs = np.asarray(values[1:], dtype='float32') 
	            embeddings_index[word] = coefs 

	    # Generating embeddings 
	embeddings_matrix = np.zeros((len(word_index ) + 1, embedding_dim)) 
	for word, i in word_index.items(): 
	        embedding_vector = embeddings_index.get(word) 
	        if embedding_vector is not None: 
	            embeddings_matrix[i] = embedding_vector

	return embeddings_matrix