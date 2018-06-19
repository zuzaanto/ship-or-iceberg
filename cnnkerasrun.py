import plot

import sys
import io
import urllib
import requests
import json
import pandas
import numpy as np
import os
import cnnkerasmodel as cnn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss



train = pandas.read_json("./data/processed/train.json")

X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
target_train=train['is_iceberg']
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)

#cross-validation training
gmodel=cnn.KerasModel()
K=3
folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=16).split(X_train, target_train))
y_test_pred_log = 0
y_train_pred_log=0
y_valid_pred_log = 0.0*target_train
for j, (train_idx, test_idx) in enumerate(folds):
    print('\n===================FOLD=',j)
    X_train_cv = X_train[train_idx]
    y_train_cv = target_train[train_idx]
    X_holdout = X_train[test_idx]
    Y_holdout= target_train[test_idx]
    
    #define file path and get callbacks
    file_path = "%s_cnn_model_weights.hdf5"%j
    callbacks = cnn.get_callbacks(filepath=file_path, patience=5)
    # gen_flow = gen_flow_for_input(X_train_cv, y_train_cv)
    gmodel.fit(X_train_cv, y_train_cv,
            batch_size=24,
            epochs=30,
            verbose=1,
            validation_data=(X_holdout, Y_holdout),
            callbacks=callbacks)
    gmodel.load_weights(filepath=file_path)
    #Getting Training Score
    score = gmodel.evaluate(X_train_cv, y_train_cv, verbose=0)
    print('Train loss:', score[0])
    print('Train accuracy:', score[1])
    #Getting Test Score
    score = gmodel.evaluate(X_holdout, Y_holdout, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    #Getting validation Score.
    pred_valid=gmodel.predict(X_holdout)
    y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])
    #Getting Train Scores
    temp_train=gmodel.predict(X_train)
    y_train_pred_log+=temp_train.reshape(temp_train.shape[0])
#y_test_pred_log=y_test_pred_log/K
y_train_pred_log=y_train_pred_log/K
print('\n Train Log Loss Validation= ',log_loss(target_train, y_train_pred_log))
print(' Test Log Loss Validation= ',log_loss(target_train, y_valid_pred_log))
