# GazeEstimation-CNN
This project is about human gaze estimation using Convolutional Neural Networks.<br/ >
The baseline method is "Appearance-Based Gaze Estimation in the Wild"<br/ >
The dataset used here is EYEDIAP.
Here I have used LeNet network too. but there are two main difference with the baseline:<br/ >
1) I have extracted heaspose information using deep learning method, i.e. I have not used vectors of 12(9:rotation matrxi,3:tranlsation matrix) direcltly,but used another LeNEt network to train and estimate these 12 values automatically.
2) I have change LeNet network,adding another layer of convolution,"conv3",before the last pooling layer,with the following configuration:<br/ >

layers {<br/ >
  name: "conv3"<br/ >
  type: CONVOLUTION<br/ >
  bottom: "conv2"<br/ >
  top: "conv3"<br/ >
  blobs_lr: 1<br/ >
  blobs_lr: 2<br/ >
  convolution_param {<br/ >
    num_output: 1<br/ >
    kernel_size: 5<br/ >
    stride: 1<br/ >
    weight_filler {<br/ >
      type: "gaussian"<br/ >
      std: 0.01<br/ >
    }<br/ >
    bias_filler {<br/ >
      type: "constant"<br/ >
    }<br/ >
  }<br/ >
}<br/ >

