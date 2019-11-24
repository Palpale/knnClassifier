#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import csv

#Read the data from a scv file and return it on a numpy array
def read_data(file):
    with open(file, 'r') as f:
        data = list(csv.reader(f, delimiter=','))
    data = np.array(data, dtype=np.float)
    return(data)

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

    for j in range(0,10):
        err_by_digits[j] = err_by_digits[j]*100/float(nbDigLab[j+1])

    return err_by_digits


if __name__ == "__main__":
    #Load data
    ind_test = read_data("../result/fullbyind.csv")
    ind_test = ind_test.astype(int)
    test_lbl = read_data("../mnist/t10k-labels.csv")
    train_lbl = read_data("../mnist/train-labels.csv")

    kmax = 50

    #compute all result and error for k 1 -> kmax
    nbImgTest = test_lbl.shape[0]
    errork = np.zeros(shape=(kmax, 1))
    errorkByDigit = np.zeros(shape=(10, kmax))

    for k in range(1, kmax + 1):
        temp = train_lbl[ind_test[:, 0:k]]
        temp = temp.astype(int)
        temp = np.squeeze(temp, axis=2) #remove 1 dimension useless

        result = np.zeros(shape=(nbImgTest, 1))

        #compute the majority class 
        for x in range(0, nbImgTest):
            result[x] = np.bincount(temp[x,:]).argmax()
        
        #compute the error
        errork[k-1] = compute_error(result, test_lbl)

        #compite the error by digit
        errorkByDigit[:,k-1] = np.squeeze(compute_error_digits(np.squeeze(result), np.squeeze(test_lbl)))

    ##Plot results
    karray = np.arange(1, kmax + 1)
    names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    #plot error for k
    plt.figure()
    plt.title("Percentage of error in function of k")
    plt.xlabel("k")
    plt.ylabel("Percentage of error")
    plt.grid(True)
    plt.plot(karray, errork)

    plt.suptitle("kNN Classifier on 10k test images with a training on 60k images")
    plt.show()
    
    #plot error by digit
    plt.figure()
    plt.title("Percentage of error for each digit for k = 3")
    plt.xlabel("digits")
    plt.ylabel("Percentage of error")
    plt.bar(names, errorkByDigit[:, 2])
    plt.suptitle("kNN Classifier on 10k test images with a training on 60k images")
    plt.show()

    #plot error for each k for each digit
    fig, ax = plt.subplots()
    plt.title("Percentage of error in function of k for each digit")
    plt.xlabel("k")
    plt.title("Percentage of error in function of k")
    plt.grid(True)
    ax.plot(karray, errorkByDigit[0,:], 'b', label='digit 0')
    ax.plot(karray, errorkByDigit[1,:], 'g', label='digit 1')
    ax.plot(karray, errorkByDigit[2,:], 'r', label='digit 2')
    ax.plot(karray, errorkByDigit[3,:], 'c', label='digit 3')
    ax.plot(karray, errorkByDigit[4,:], 'm', label='digit 4')
    ax.plot(karray, errorkByDigit[5,:], 'y', label='digit 5')
    ax.plot(karray, errorkByDigit[6,:], 'k', label='digit 6')
    ax.plot(karray, errorkByDigit[7,:], 'b--', label='digit 7')
    ax.plot(karray, errorkByDigit[8,:], 'g--', label='digit 8')
    ax.plot(karray, errorkByDigit[9,:], 'r--', label='digit 9')
    legend = ax.legend(loc='upper right')

    plt.suptitle("kNN Classifier on 10k test images with a training on 60k images")
    plt.show()
