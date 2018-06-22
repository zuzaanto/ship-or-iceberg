# Ship Or Iceberg
This is a project of 4 different machine learning solutions created in correspondance with a Kaggle competition, https://www.kaggle.com/c/statoil-iceberg-classifier-challenge.
The data used in this project can be downloaded after acceptance of regulations through the given website.

The data consists of 2-channel satellite images; each one contains an unidentified object at sea. The training set (the only set labelled) contained 1604 examples.

The full paper on these solutions is a BSc thesis created for WUT, and if interested, can be obtained upon request. 

# CNN

The best solution proved to be the CNN created using the Keras library, with accuracy of over 0.9 on test, and 0.91 on train set.

In file cnnkerasmodel there is the network model, created using Keras, using Tensorflow back-end. It is a conventional CNN of 8 layers. Training is specified in file cnnkerasrun. It uses cross-validation (test accuracy obtained without cross-validation - by dividing the set at the start into training and test - was smaller by ca. 1%, which proves that cross-validation accuracy is trustworthy). Training 90 epochs took ca. 1-2h on an average modern CPU.

# Transfer-learning

The second-best solution was the use of pretrained model VGG16. This method also took advantage of Keras library, and SciKit-Learn for ex. cross-validation. The last layer of VGG16 was removed and replaced with 3 new layers. Training 100 epochs took ca. 8h on that same CPU. The accuracy was ca. 0.85 on test set, and 0.88 on train set. 
# SVM

The last two solutions were based on an SVM, with normalized pixel values (1) or with feature extraction using HOG description algorithm in OpenCV library. The HOG extraction proved to be thoroughly unnecessary due to the data character (background values were already near 0, and objects were positive values), therefore the accuracy was near random. However, use of simple normalization proved efficient, and accuracy on test set was 0.82, while on train - 0.9. This suggests that SVM may have been overfitted. Of course, the SVM models where created with use of GridSearch function from SciKit-Learn, which helped determine optimal C and gamma values. 

# 

In "plot" file can be found a function based on Plotly library, which helps visualise data points in 3d. Visual classification of data points is a valid test for machine learning problems, since neural networks are (loosely) based on the human brain, after all.
#


Copyright 2018 Zuzanna Szafranowska

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.