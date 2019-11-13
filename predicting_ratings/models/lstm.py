# lstm.py
#
# Anders Poirel
# 13-11-2019

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Embedding, LSTM, Bidirectional

def lstm_model(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'], num_units = 128, embedding_size = 16):
<<<<<<< HEAD
    """

    Returns: (keras Model) compiled keras model

    Defines a bidirectional LSTM model with embeddings.
    """
=======
>>>>>>> 32c08444e7ee84cd6fe47d3f5a72dc8206ae8a0b

    model = tf.keras.Sequential()
    model.add(Embedding(input_dim = 10000, output_dim = embedding_size, input_length = 1050))
    model.add(Bidirectional(LSTM(num_units)))
    model.add(Dense(1, activation = 'softmax'))
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
<<<<<<< HEAD

=======
    
>>>>>>> 32c08444e7ee84cd6fe47d3f5a72dc8206ae8a0b
    return model