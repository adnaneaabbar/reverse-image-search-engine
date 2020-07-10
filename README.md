# Creating an image classification search engine using CNNs 

This project's main purpose is to mimic Google's Image Search Engine, working with a large enough dataset we will try to achieve the best results we can.

---

First you should load the dataset from Darknet : [Here](https://pjreddie.com/projects/cifar-10-dataset-mirror/)

Second, you should put in the root of your project following this structure :


```
. project
+-- your_notebook.ipynb
+-- dataset
|   +-- train       ==> this contains your 50.000 training images
|   +-- test        ==> this contains your 10.000 training images
|   +-- labels.txt  ==> this contains your classes

```
---

# Metrics used to compared query vector to training vector:

## 1. Cosine similarity:

<img src="https://github.com/adnaneaabbar/reverse-image-search-engine/blob/master/static/cosine-similarity-draw.png" width="600" height="300">

<img src="https://github.com/adnaneaabbar/reverse-image-search-engine/blob/master/static/cosine-similarity.png" width="600" height="300">

## 2. Hamming distance:

<img src="https://github.com/adnaneaabbar/reverse-image-search-engine/blob/master/static/hamming-explained.png" width="600" height="300">

<img src="https://github.com/adnaneaabbar/reverse-image-search-engine/blob/master/static/hamming.png" width="600" height="300">

# Model summary

The entire model consists of 14 layers in total. In addition to layers below lists what techniques are applied to build the model.

<img src="https://github.com/adnaneaabbar/reverse-image-search-engine/blob/master/static/conv_model.png">


1. Convolution with 64 different filters in size of (3x3)
2. Max Pooling by 2
* ReLU activation function
* Batch Normalization
3. Convolution with 128 different filters in size of (3x3)
4. Max Pooling by 2
* ReLU activation function
* Batch Normalization
5. Convolution with 256 different filters in size of (3x3)
6. Max Pooling by 2
* ReLU activation function
* Batch Normalization
7. Convolution with 512 different filters in size of (3x3)
8. Max Pooling by 2
* ReLU activation function
* Batch Normalization
9. Flattening the 3-D output of the last convolutional operations.
10. Fully Connected Layer with 128 units
* Dropout
* Batch Normalization
11. Fully Connected Layer with 256 units
* Dropout
* Batch Normalization
12. Fully Connected Layer with 512 units
* Dropout
* Batch Normalization
13. Fully Connected Layer with 1024 units
* Dropout
* Batch Normalization
14. Fully Connected Layer with 10 units (number of image classes)

# Deploying using Flask


# Reference

[CIFAR10](https://github.com/deep-diver/CIFAR10-img-classification-tensorflow)
