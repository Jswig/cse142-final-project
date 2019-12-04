# generate_predictions.py

# Anders Poirel
# 04-12-2019


import numpy as np
from lstm import lstm_model

CHECKPOINT_PATH = '../../output/checkpoint.ckpt'
DATA_PATH = '../../data/processed/'

model = lstm_model()
model.load_weights(CHECKPOINT_PATH)

X_test = np.load(DATA_PATH + 'X_test.npy')
y_pred = model.predict(X_test)

# predictions post-processing
row_maxes = y_pred.max(axis = 1).reshape(-1, 1)
y_pred[:] = np.where(y_pred == row_maxes, 1, 0)

y_pred = pd.DataFrame(y_pred, columns = ['1', '2', '3', '4', '5'])
y_pred = y_pred.idxmax(axis = 1).values.astype(np.int)

submission_df = pd.DataFrame({'Predictions' : y_pred})
submission_df.to_csv('predictions.csv',index = False)