# lstm_preporcessing.py
#
# Anders Poirel
# 12-11-2019

import sys
import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequence
from pandas import get_dummies


def extract_train(raw_path, processed_path, max_features):
    """
    Parameters:
    - raw_path: (String) path of the training set in .json format
    - processed_path: (String) path where to save the processed data
    - max_features: (int) maximum number of different words created by the tokenizer

    Extracts data from the json, saving the results to a serialized file.
    For now, this function only extracts text and ratinds, discarding the rest
    """

    try:
        dataset_f = open(path, 'r')
        
    except OSError as err:
        print('OS error: {0}'.format(err))
        sys.exit(1)

    else:
        dataset = json.load(dataset_f)

        ratings = [item['stars'] for item in dataset]
        reviews =[item['text'] for item in dataset]

        tokenizer = Tokenizer(max_features)
        tokenizer.fit_on_texts(reviews)
        reviews_t = tokenizer.texts_to_sequences(reviews)
        ratings_t = get_dummies(ratings)

        np.save(path + '/X_train.npy', reviews_t)
        np.save(path + '/y_train.npy', ratings_t)

        dataset_f.close()


if __name__ == "main":

    # FIXME: add functions to automatically download and extract dataset
    # to appropirate folder. Also, add/adapt function for also handling 
    # training data

    pass


