# GazeEstimation-CNN
This project is about human gaze estimation using Convolutional Neural Networks.<br/ >
The baseline method is "Appearance-Based Gaze Estimation in the Wild"<br/ >
The dataset used here is EYEDIAP.
Here I have used LeNet network too. but there are two main difference with the baseline:<br/ >
1) I have extracted heaspose information using deep learning method, i.e. I have not used vectors of 12(9:rotation matrxi,3:tranlsation matrix) direcltly,but used another LeNEt network to train and estimate these 12 values automatically.
2) I have change LeNet network,adding another layer of convolution.
