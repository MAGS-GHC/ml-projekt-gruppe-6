import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow.compat.v1 as tf

from tensorflow.keras.models import load_model

from data_processing import preprocess_data, encode_labels
from text_processing import tokenize_and_sequence, generate_word_embeddings
from model_building import build_model


def main():
	data = pd.read_csv("./files/news.csv")
	data = preprocess_data(data)
	data = encode_labels(data)

	padding_type = 'post'
	trunc_type = 'post'
	test_portion = 0.1
	embedding_dim = 50
	training_size = 3000
	
	
	max_length = 54
	#oov_tok = "<OOV>"

	num_epochs = 50
	
	tokenizer, word_index, vocab_size, training_sequences, test_sequences, test_labels, training_labels = tokenize_and_sequence(data, padding_type, trunc_type, test_portion, training_size)
	
	embeddings_matrix = generate_word_embeddings(word_index, embedding_dim)
	
	#build_model(vocab_size, embedding_dim, max_length, embeddings_matrix, training_sequences, training_labels, test_sequences, test_labels, num_epochs)
	
	
	Y = ["Russia is now war with Ukraine", "Donald Trump is now a president of The United States", "Hillary Clinton is now a the president of the United States"]


	sequences = tokenizer.texts_to_sequences([Y[1]])[0]

	sequences = pad_sequences([sequences], maxlen=max_length, 
						padding=padding_type, 
						truncating=trunc_type)

	
	loaded_model = load_model('files/model.keras')

	if(loaded_model.predict(sequences, verbose=0)[0][0] >= 0.5):
		print("This news is True")
	else: 
		print("This news is false")

if __name__ == "__main__":
    main()