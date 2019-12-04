# lstm.py
#
# Anders Poirel
# 13-11-2019

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras import regularizers

def lstm_model(optimizer = 'adam', loss = 'categorical_crossentropy', 
               metrics = ['accuracy'], num_units = 128,
               embedding_size = 64, dropout = 0.2):
    """

    Returns: (keras Model) compiled keras model

    Defines a bidirectional LSTM model with embeddings.
    """

    model = tf.keras.Sequential()
    model.add(Embedding(input_dim = 10000, output_dim = embedding_size,
                        input_length = 1050))
    model.add(Bidirectional(LSTM(num_units,
      dropout = dropout, recurrent_dropout = dropout,
      # kernel_regularizer = regularizers.l1(0.01),
      recurrent_regularizer = regularizers.l1(0.01),
      # activity_regularizer = regularizers.l1(0.01),
      bias_regularizer = regularizers.l1(0.01))))
    model.add(Dense(5, activation = 'softmax'))
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    
    return model

