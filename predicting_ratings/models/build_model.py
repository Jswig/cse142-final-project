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

sns.set()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    return

def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower right')
    return

checkpoint_path = 'output/weights/1'

X_train = np.load('../../data/processed/X_train.npy')
y_train = np.load('../../data/processed/X_train.npy')

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, save_weights_only = True, verbose = 1)

model = lstm_model()

history = model.fit(X_train, y_train, validation_split = 0.2, epochs = 10, workers = -1, use_multiprocessing = True, callbacks = [cp_callback])
    
# saves training curves
plot_acc(history)
plt.savefig('../../output/acc1.png')

plot_loss(history)
plt.savefig('../../output/loss1.png')

# saves two things:
# - entire model so that it can be reloaded for testing. Use 
# tf.keras.models.load_model to load it
# - model weights for initializing weights in future models with same weights. Use model.load_weights to load it 
model.save_weights('../../output/weights/1', save_format = 'tf')
model.save('../../output/models/model1.h5')
