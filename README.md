# Bayesian_HandwrittenDigits_MNIST
Bayesian Approach to recognize the Handwritten digits ( from MNIST )

Overview
These scripts use the MNIST image dataset of handwritten digits (0 to 255 grey scale). 
source: http://yann.lecun.com/exdb/mnist/

The dataset is divided into two parts for training (60,000 counts) and test (10,000 counts).
path: .\\orig_data\\mnist_train.csv & .\\orig_data\\mnist_test.csv

Preprocessing on this dataset was performed by the deskew operation (using the script from the following source with little modifications).
source: https://fsix.github.io/mnist/Deskewing.html
path: .\\mod_data\\deskewdata_train.csv & .\\mod_data\\deskewdata_test.csv

Training data is used to calculate the covariance values for each of the pixels (0 to 783).
For each of the digits (0 to 9), the mean value is calculated for each of the 784 pixels.

To identify the value of digits from the test dataset following approach is used.
Using the 784 pixels values, the logarithm  pdf (probability density value) is calculated for each digit (0-9). The digit which has the largest log pdf value is predicted as the digit for those pixel values. The probability distribution is used as Multi-variate normal with mean and covariance values used from the previous step of training images.

This Bayesian  approach yields accuracy of 91%.
.\\results\\...