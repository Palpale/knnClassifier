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
def knn_classifier(train_img, train_lbl, test_img, k):
    nbImgTest = test_img.shape[0]
    nbImgTrain = train_img.shape[0]

    bar = Bar('Processing', max=nbImgTest)

    norm = np.zeros(shape=(nbImgTest, nbImgTrain))
    result = np.zeros(shape=(nbImgTest, 1))

    for i in range(0, nbImgTest):
        bar.next()
        for j in range(0, nbImgTrain):
            norm[i,j] = euclidean_dist_img(train_img[j,:], test_img[i,:])

    ind  = np.argsort(norm, axis=1) #sort norm by indices

    #remove comment if you want to save the indices
    #np.savetxt("../result/fullbyind.csv", ind, delimiter=',')


    temp = train_lbl[ind[:, 0:k]] #take the k first indice (k neirest images)
    temp = temp.astype(int)
    temp = np.squeeze(temp) #remove 1 dimension useless

    for x in range(0, nbImgTest):
        result[x] = np.bincount(temp[x,:]).argmax()

    return result
    
def compute_error(result, labels):
    return np.mean(result != labels)*100


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
    final = knn_classifier(training_set_images, training_set_labels, test_set_images[1:200,:], k)
    np.savetxt("../result/resultk" + str(k) + ".csv", final, delimiter=',')

    #compute the error
    err = compute_error(final, test_set_labels[1:200])
    print err