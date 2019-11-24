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
    
#KNN classifier, compute the error if we give the labels of test_img
def knn_classifier(train_img, train_lbl, test_img, k, **kwargs):
    nbImgTest = test_img.shape[0]
    nbImgTrain = train_img.shape[0]

    for key, value in kwargs.items():
        if(key == 'test_lbl'):
            check = True
            test_lbl = value

    #Errors detection
    if(test_img.shape[1] != train_img.shape[1]):
        print("Error 1 train and test don't have same dimension")
        exit(1)
    if(train_lbl.shape[0] != nbImgTrain):
        print("Error 2 number of label must coinced with number of train images")
        exit(1)
    if(k <= 0):
        print("Error 3 k must be > 1")
        exit(1)
    if(check and (test_lbl.shape[0] != test_img.shape[0])):
        print("Error 4 number of label must coinced with number of test images")
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
    temp = np.squeeze(temp, axis=2) #remove 1 dimension useless

    #compute the majority class 
    for x in range(0, nbImgTest):
        result[x] = np.bincount(temp[x,:]).argmax()

    #compute the error
    if(check):
        err = compute_error(result, test_lbl)
        print("error (%) = " + str(err))

    return result

#Compute the error (%) between the prediction and the reality
def compute_error(result, labels):
    if(result.shape != labels.shape):
        print("Error result and labels don't have the same dimension")
        exit(1)

    return np.mean(result != labels)*100

#Compute the error (%) for each digits
def compute_error_digits(result,labels):
    if(result.shape != labels.shape):
        print("Error result and labels don't have the same dimension")
        exit(1)

    labels = labels.astype(int)
    nbDigLab = np.bincount(labels)

    err_by_digits = np.zeros(shape=(10, 1))

    for i in range(0, labels.shape[0]):
        if(result[i] != labels[i]):
            err_by_digits[labels[i]-1] += 1

    print(err_by_digits)

    for j in range(0,10):
        err_by_digits[j] = err_by_digits[j]*100/float(nbDigLab[j+1])

    return err_by_digits

#Compute the euclidean distance between two images
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

    NbDataTest = 10

    #compute the knn classifier
    final = knn_classifier(training_set_images, training_set_labels, test_set_images[0:NbDataTest,:], k, test_lbl=test_set_labels[0:NbDataTest,:])
    #np.savetxt("../result/resultk" + str(k) + ".csv", final, delimiter=',')

    #compute the error by digit
    dig_error = compute_error_digits(np.squeeze(final), np.squeeze(test_set_labels[0:NbDataTest,:]))
    print(dig_error)

    #plot the error
    names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    plt.figure()
    plt.bar(names, np.squeeze(dig_error))
    plt.title("Error by digit ( k = " + str(k) + " , nb data tested : " + str(NbDataTest) + " )")
    plt.show()