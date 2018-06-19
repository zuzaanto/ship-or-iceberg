import os
# Importing packages:
import numpy as np
import pandas as pd

from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score,f1_score,log_loss,roc_auc_score, classification_report

from sklearn.model_selection import train_test_split,ShuffleSplit,cross_val_score,GridSearchCV

from sklearn import svm
from sklearn.linear_model import LogisticRegression,SGDClassifier

from os.path import join as opj
from matplotlib import pyplot as plt

from matplotlib.colors import Normalize

# Reading the traning data set json file to a pandas dataframe
train=pd.read_json('./data/processed/train.json')


# Replace the 'na's with numpy.nan
train.inc_angle.replace('na', np.nan, inplace=True)

# Drop the rows that has NaN value for inc_angle
train.drop(train[train['inc_angle'].isnull()].index,inplace=True)
print("replacing NaN inc angles")

#train.head(5)

X_HH_train=np.array([np.array(band).astype(np.float32) for band in train.band_1])
X_HV_train=np.array([np.array(band).astype(np.float32) for band in train.band_2])
X_angle_train=np.array([[np.array(angle).astype(np.float32) for angle in train.inc_angle]]).T
y_train=train.is_iceberg.values.astype(np.float32)
X_train=np.concatenate((X_HH_train,X_HV_train,X_angle_train), axis=1)
X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, y_train, random_state=1, train_size=0.75)

# Now, we have 75*75 numerical features for band_1, 75*75 numerical features for band_2, and 1  feature for angle 

scaler = MaxAbsScaler()
X_train_maxabs = scaler.fit_transform(X_train_cv)
scaler = MaxAbsScaler()
X_valid_maxabs = scaler.fit_transform(X_valid)
# Create the SVM instance using Radial Basis Function (rbf) kernel
clf = svm.SVC(kernel='rbf',probability=False)
# Set the range of hyper-parameter we wanna use to tune our SVM classifier
C_range = [0.1,1,10,50,100]
gamma_range = [0.00001,0.0001,0.001,0.01,0.1]
param_grid_SVM = dict(gamma=gamma_range, C=C_range)
# set the gridsearch using 3-fold cross validation and 'ROC Area Under the Curve' as the cross validation score. 
grid = GridSearchCV(clf, param_grid=param_grid_SVM, cv=3,scoring='roc_auc', verbose = 10)
grid.fit(X_train_maxabs, y_train_cv)
print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

y_true, y_pred = y_valid, grid.predict(X_valid_maxabs)
print("Detailed classification report:")
print(classification_report(y_true, y_pred))
