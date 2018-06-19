# Importing packages:
import numpy as np
import pandas as pd

from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score,f1_score,log_loss,roc_auc_score

from sklearn.model_selection import train_test_split,ShuffleSplit,cross_val_score,GridSearchCV

from sklearn import svm
from sklearn.linear_model import LogisticRegression,SGDClassifier

from os.path import join as opj
from matplotlib import pyplot as plt
import time
import cv2

from matplotlib.colors import Normalize

def _HOG(images):
    WIN_SIZE = (75, 75)
    BLOCK_SIZE = (15, 15)
    BLOCK_STRIDE = (6, 6)
    CELL_SIZE = (5, 5)
    NBINS = 9
    hog_desriptor = cv2.HOGDescriptor(WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS)
    hog_features = [np.squeeze(hog_desriptor.compute(images[idx])) for idx in range(images.shape[0])]
    return np.stack(hog_features, axis=0)

def generate_batch(images, labels, batch_size):
    indices = np.arange(images.shape[0])
    np.random.shuffle(indices)
        
    for batch_idx in range(images.shape[0] // batch_size):
            
        batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_images = images[batch_indices, ...]
        batch_labels = labels[batch_indices]
        yield batch_images, batch_labels

def resize_imgs(images):
    scaling_factor=32./75.
    small_imgs = np.zeros((images.shape[0], 32, 32))
    # small_imgs = []
    for img in range(images.shape[0]):
        small_img=cv2.resize(images[img],(32,32))
        # small_img=cv2.resize(images[img, ...],None,fx=scaling_factor,fy=scaling_factor,interpolation=cv2.INTER_AREA)
        small_imgs[img]=small_img
    return small_imgs
def extract_HOG(images, labels, verbose=True):
    samples = images.shape[0]
    for images, labels in generate_batch(images=images, labels=labels, batch_size=samples):
        start_time = time.time()
        features = _HOG(images)
        hog_time = time.time()
        if verbose:
            print("Features calculated in ", hog_time - start_time, " seconds")
        return features, labels
#funkcja Daniela:
def classify_svm(features_train, labels_train, features_test, labels_test, C, verbose):
    svm_time_start = time.time()
#    classifier_svm = LinearSVC(C=C, verbose=verbose, dual=False, max_iter=5000)
    classifier_svm = svm.LinearSVC(C=C, verbose=verbose, dual=True, max_iter=5000)
    classifier_svm.fit(features_train, labels_train)
    svm_time_fit = time.time()
    print( "SVM fit in ", svm_time_fit - svm_time_start, " seconds\n\n")
    print( "TRAIN SCORE = ", classifier_svm.score(features_train, labels_train))
    print( "TEST  SCORE = ", classifier_svm.score(features_test, labels_test))


# Reading the traning data set json file to a pandas dataframe
train=pd.read_json('./data/processed/train.json')

# Replace the 'na's with numpy.nan
train.inc_angle.replace('na', np.nan, inplace=True)

# Drop the rows that has NaN value for inc_angle
train.drop(train[train['inc_angle'].isnull()].index,inplace=True)
print("replacing NaN inc angles")

X_HH_train=np.array([np.array(band).astype(np.float32) for band in train.band_1])
X_HV_train=np.array([np.array(band).astype(np.float32) for band in train.band_2])
X_angle_train=np.array([[np.array(angle).astype(np.float32) for angle in train.inc_angle]]).T
y_train=train.is_iceberg.values.astype(np.float32)
X_lol=np.concatenate((X_HH_train,X_HV_train), axis=1)
print("SHAPEEEEEEE",X_lol.shape)
#normalizacja przed ekstrakcją cech
max_hh=np.amax(X_HH_train)
min_hh=np.amin(X_HH_train)
max_hv=np.amax(X_HV_train)
min_hv=np.amin(X_HV_train)
X_HH_train_norm = (X_HH_train - min_hh)/(max_hh-min_hh)
X_HV_train_norm = (X_HV_train - min_hv)/(max_hv-min_hv)
print("after first normalization, maximum: ",np.amax(X_HH_train_norm),", minimum:", np.amin(X_HH_train_norm))

X_HH_train_norm = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in X_HH_train_norm])
X_HV_train_norm = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in X_HV_train_norm])
#resizing images, wrrrrr
# X_HH_train_norm = np.array(resize_imgs(X_HH_train_norm)*255, dtype=np.uint8)
# X_HV_train_norm = np.array(resize_imgs(X_HV_train_norm)*255, dtype=np.uint8)
#druga normalizacja
X_HH_train_norm = np.array(X_HH_train_norm*255, dtype=np.uint8)
X_HV_train_norm = np.array(X_HV_train_norm*255, dtype=np.uint8)

print("after second normalization, maximum: ",np.amax(X_HH_train_norm),", minimum:", np.amin(X_HH_train_norm))

#ekstrakcja cech dla obu obrazów:
features_HH, labels_HH = extract_HOG(X_HH_train_norm, y_train)
features_HV, labels_HV = extract_HOG(X_HV_train_norm, y_train)


print("shape of features hh and hv", features_HH.shape, features_HV.shape)
print("shape of labels HH and HV", labels_HH.shape, labels_HV.shape)
print("shape of regular hh, hv, and y", X_HH_train_norm.shape, X_HV_train_norm.shape, y_train)
#labels_f =  np.concatenate((labels_HH[:,np.newaxis], labels_HV[:, np.newaxis]), axis=-1)
X_HH_train_norm_flat = np.array([X_HH_train_norm[idx].flatten() for idx in range(X_HH_train_norm.shape[0])])
X_HV_train_norm_flat = np.array([X_HV_train_norm[idx].flatten() for idx in range(X_HV_train_norm.shape[0])])
print("shape of flattened train data:", X_HH_train_norm_flat.shape, X_HV_train_norm_flat.shape)
X_train=np.concatenate((X_HH_train_norm_flat[:,:,np.newaxis], X_HV_train_norm_flat[:,:,np.newaxis]), axis=-1)
X_features = np.concatenate((features_HH[:,:,np.newaxis],features_HV[:,:,np.newaxis]),axis=-1)
X_features_flattened = np.array([X_features[idx].flatten() for idx in range(X_features.shape[0])])
print("shape of features ",X_features.shape)
print("shape of flattened features ",X_features_flattened.shape)
print("shape of conc flattened data: ", X_train.shape)
print("shape of y_train: ",y_train.shape)

#podział zbioru na trening i walidacje (opcjonalnie, bo i tak dalej jest cross walidacja)
X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, y_train, random_state=1, train_size=0.75)
X_features_cv, X_f_valid, y_features_cv, y_f_valid = train_test_split(X_features_flattened, y_train, random_state=1, train_size=0.75)

clf = svm.SVC(kernel='rbf',probability=False)
# Set the range of hyper-parameter we wanna use to tune our SVM classifier
C_range = [0.1,1,10,50,100]
gamma_range = [0.00001,0.0001,0.001,0.01,0.1]
param_grid_SVM = dict(gamma=gamma_range, C=C_range)
# set the gridsearch using 3-fold cross validation and 'ROC Area Under the Curve' as the cross validation score. 
grid = GridSearchCV(clf, param_grid=param_grid_SVM, cv=3,scoring='roc_auc', verbose = 10)
grid.fit(X_features_cv, y_features_cv)
print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

y_true, y_pred = y_f_valid, grid.predict(X_f_valid)
print("Detailed classification report:")
print(classification_report(y_true, y_pred))
