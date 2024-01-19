import tensorflow.compat.v1 as tf
import numpy as np

def build_model(vocab_size, embedding_dim, max_length, embeddings_matrix, training_sequences, training_labels, test_sequences, test_labels, num_epochs):
	model = tf.keras.Sequential([ 
	tf.keras.layers.Embedding(vocab_size +1, embedding_dim, 
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
	


	training_padded = np.array(training_sequences) 
	training_labels = np.array(training_labels) 
	testing_padded = np.array(test_sequences) 
	testing_labels = np.array(test_labels) 
	history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels),
    verbose=2)
	model.save('files/model.keras')
