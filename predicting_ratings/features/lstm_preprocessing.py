# lstm_preporcessing.py
#
# Anders Poirel
# 12-11-2019

# tokenizes data from trainging and test sets

import sys
import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pandas import get_dummies

def extract_from_json(raw_path, processed_path, max_features, 
                      is_train_set = True):                                  
    """
    Parameters:
    - raw_path: (String) path of the training set in .json format
    - processed_path: (String) path where to save the processed data
    - max_features: (int) maximum number of different words created by the tokenizer

    Extracts data from the json, saving the results to a serialized file.
    For now, this function only extracts text and ratinds, discarding the rest
    """

    MAX_LEN = 1050

    try:
        dataset_f = open(raw_path, 'r')
        
    except OSError as err:
        print('OS error: {0}'.format(err))
        sys.exit(1)

    else:
        dataset = json.load(dataset_f)
        reviews =[item['text'] for item in dataset]

        tokenizer = Tokenizer(max_features)
        tokenizer.fit_on_texts(reviews)
        reviews_t = tokenizer.texts_to_sequences(reviews)
        reviews_t = pad_sequences(reviews_t, maxlen = MAX_LEN)
        
        if is_train_set:
            ratings = [item['stars'] for item in dataset]
            ratings_t = get_dummies(ratings)
            np.save(processed_path + '/X_train.npy', reviews_t)
            np.save(processed_path + '/y_train.npy', ratings_t)

        else:
            np.save(processed_path + '/X_test.npy', reviews_t)

        dataset_f.close()

extract_from_json('../../data/raw/data_train.json', '../../data/processed', max_features = 10000)
extract_from_json('../../data/raw/data_test_wo_label.json', '../../data/processed', max_features = 10000, is_train_set = False)