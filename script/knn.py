#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
import csv

#Read the data from a scv file and return it on a numpy array
def read_data(file):
    with open(file, 'r') as f:
        data = list(csv.reader(f, delimiter=','))
    data = np.array(data, dtype=np.float)
    return(data)
    
#KNN classifier
def knn_classifier(train_img, train_lbl, test_img, k, test_lbl=None):
    nbImgTest = test_img.shape[0]
    nbImgTrain = train_img.shape[0]

    #Errors detection
    if(test_img.shape[1] != train_img.shape[1]):
        print("Error train and test don't have same dimension")
        exit(1)
    if(train_lbl.shape[0] != nbImgTrain):
        print("Error number of label must coinced with number of train images")
        exit(1)
    if(k <= 0):
        print("Error k must be > 1")
        exit(1)
    if(test_lbl and (test_lbl.shape[0] != train_lbl.shape[0])):
        print("Error number of label must coinced with number of test images")
        exit(1)

        
    #progress bar to know where the programm is
    bar = Bar('Processing', max=nbImgTest)

    norm = np.zeros(shape=(nbImgTest, nbImgTrain))
    result = np.zeros(shape=(nbImgTest, 1))

    for i in range(0, nbImgTest):
        bar.next()
        for j in range(0, nbImgTrain):
            norm[i,j] = euclidean_dist_img(train_img[j,:], test_img[i,:])

    bar.finish()
    ind  = np.argsort(norm, axis=1) #sort norm by indices

    #remove comment if you want to save indices (for kmax = 100)
    #np.savetxt("../result/fullbyind.csv", ind[:,0:100], delimiter=',')


    temp = train_lbl[ind[:, 0:k]] #take the k first indice (k neirest images)
    temp = temp.astype(int)
    temp = np.squeeze(temp) #remove 1 dimension useless

    for x in range(0, nbImgTest):
        result[x] = np.bincount(temp[x,:]).argmax()

    #compute the error
    if(test_lbl):
        err = compute_error(result, test_lbl)
        print "error (%) = " + str(err)

    return result
    
def compute_error(result, labels):
    return np.mean(result != labels)*100

def compute_error_digits(result,labels):
    err_by_digits = np.zeros()

def euclidean_dist_img(img1, img2):
    if(img1.shape != img2.shape):
        print("Error : images must have the same size")
        return(-1)
    
    norm = np.sqrt(np.sum((img1-img2)*(img1-img2)))

    return norm


if __name__ == "__main__":
    #Load data
    training_set_images = read_data("../mnist/train-images.csv")
    training_set_labels = read_data("../mnist/train-labels.csv")
    test_set_images = read_data("../mnist/t10k-images.csv")
    test_set_labels = read_data("../mnist/t10k-labels.csv")

    #set k
    k = 3

    #compute the knn classifier
    final = knn_classifier(training_set_images, training_set_labels, test_set_images[0:10000,:], k)
    np.savetxt("../result/resultk" + str(k) + ".csv", final, delimiter=',')