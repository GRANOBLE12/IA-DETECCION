# GTSRB Challenge

a.k.a German Traffic Sign Recognition Benchmark :de: :no_entry: :no_bicycles:
:no_entry_sign: ...

## Goal

Use [Torch](http://torch.ch/) to train and evaluate a 2-stage convolutional
neural network able to classify German traffic sign images (43 classes):

* fork the repository under your account,
* go to Settings > Features and enable Issues,
* create an issue under your repo describing your approach,
* report your result(s),
* commit your code,
* edit the README with pre-requisites and usage,
* boost accuracy by experimenting the multi-scale architecture,
* compare with the results obtained in matching mode (i.e use the features with a distance-based search).

## Paper

[Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://computer-vision-tjpn.googlecode.com/svn/trunk/documentation/reference_papers/2-sermanet-ijcnn-11-mscnn.pdf), by Yann LeCun et al.

## Dataset

### Training

`http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip` (263 MB)

### Testing

`http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip` (84 MB)
`http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip` (98 kB)

# Presentation of the solution

## Disclaimer
This projec is not completely finished yet.
Some part of the code should be refactored and command line argument should be used.

## Usage

* Data

Download the data and decompress them in the GTSRB at the root folder of this project.
There should be the Final_Training and Final_Test folders in THIS_REPO/GTSRB/
The Final_training does no need to be modified. For the Final_Test, you need to put the content of Final_Test/Images/* in Final_Test/Images/final_test/* . You also need to extract the ground truth csv and put it in Final_Test/Images/final_test/GT-final_test.csv .

* Run the code

You can use the main module with ```time th main.lua``` .
All the parameters are directly specified in the corresponding module.

## Modules

The solution is separated in 5 modules:

* dataset
    * read the dataset
    * perform the normalization
    * perform jittering

* network
    * create the torch module representing the network

* training or training_optim
    * perform the training of a torch module using either nn.StochasticGradient or the torch.optim module

* testing
    * score the given torch module with the given dataset

* main
    * combine all the modules to perform the end to end training
    * cache the datasets and results in .bin files.

## Optimization module

### nn.StochasticGradient

This module is very easy to use and to train.
It perform stockastic gradient descent.

The following parameters can be changed:

* learning rate
* learning rate decay
* number of epochs

The following parameters cannot be changed:

* batch size is 1
* weight decay is 0
* momentum is 0

### torch.optim

This module is a complete optimization module that contains other optimization than SGD. Here for example, we can use SGD, CG and LBFGS.

All parameters can be changed by the user depending on the optimization method you want to use

To be able to use this module, we linearize all the parameters and the gradient of the network into a set of parameters.
We then create a function that, given a set of parameters  return the loss and its gradients on a batch of the images.
The module is then doing the update of the parameters using one of the methods listed above.
This step is repeated for all batches and the desired number of epochs.

# IA-DETECCION
# IA-DETECCION
# IA-DETECCION
