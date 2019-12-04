# build_model.py
#
# Anders Poirel
# 14-11-2019

#%%
import numpy as np
import tensorflow as tf
from lstm import lstm_model
import seaborn as sns
import matplotlib.pyplot as plt


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    return

def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower right')
    return

CHECKPOINT_PATH = '../../output/checkpoint.ckpt''
DATA_PATH = '../../data/processed/'

X_train = np.load(DATA_PATH + 'X_train.npy')
y_train = np.load(DATA_PATH + 'y_train.npy')

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                              min_delta = 0.001,
                              patience = 3,
                              verbose = 0, mode = 'auto')


model = lstm_model()
history = model.fit(X_train, y_train, validation_split = 0.2, epochs = 10, workers = -1, use_multiprocessing = True, callbacks = [cp_callback])
    

# saves training curves
sns.set()

plot_acc(history)
plt.savefig('../../output/acc1.png')

plot_loss(history)
plt.savefig('../../output/loss1.png')
