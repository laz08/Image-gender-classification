#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import getopt
import time

import Utils
import ClassifierManager as cm

PERFORM_KNN = False
PERFORM_SVM = False
PERFORM_MLP = False
PERFORM_RAND_FOREST = False

PERFORM_CROSS_KNN = False
PERFORM_CROSS_SVM = False


def usage():

    print ('Usage: main.py -i <inputfile>')
    sys.exit(2)


def main(argv):

    labels = []
    if len(argv) < 1:
         usage()
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        usage()

    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ("-i", "--ifile"):
            filein = arg
        elif opt in ("-s", "--source"):
            filein = arg

    print ("[1] Reading from file " + filein)
    mat = Utils.readAsMatrix(filein)


    #############################################
    imagesNumberPerGender = 3000   # If the num. of images is greater than the maximum of them, then it reads all of them
    print("[2] Forcing gender parity up to {} images per gender".format(imagesNumberPerGender))
    mat = Utils.forceGenderParityUpToN(mat, imagesNumberPerGender)
    #############################################

    print("[3] Extracting features...")
    mat = Utils.readImages(mat)

    print("[4] Splitting on training/test set...")
    training, test = Utils.splitTrainingTestSet(mat, 0.8, True)

    if PERFORM_KNN:

        print("[5] Method chosen: KNN")
        startTime = time.time()

        if(PERFORM_CROSS_KNN):
            negLossArray = []
            diffK = [3, 5, 7, 13, 21, 51, 100]
            for k in diffK:
                negLoss = cm.performCrossvalidationKNN(mat, k)
                negLossArray.append(negLoss)
        
        else:
                cm.performKNeighbors(training, test, k)


        Utils.appendLineToFile("KnnNegLoss.csv", "K ; Log Loss")
        for k, l in enumerate(negLossArray):
            Utils.appendLineToFile("KnnNegLoss.csv", str(diffK[k]) + "; " + str(l))
        #print("Accuracy with Knn: {}%".format(acc))

        endTime = time.time()
        elapsedTime = endTime - startTime
        print("Elapsed time: {} s".format(round(elapsedTime, 4)))

    if PERFORM_SVM:

        print("[5] Method chosen: SVM")
        startTime = time.time()

        acc = cm.performLinearSVC(training, test)
        #print("SVM accuracy: {}%".format(acc))

        endTime = time.time()
        elapsedTime = endTime - startTime
        print("Elapsed time: {} s".format(round(elapsedTime, 4)))

    if PERFORM_MLP:

        print("[5] Method chosen: MLP")
        startTime = time.time()

        acc = cm.performMLPClassifier(training, test)
        print("MLP accuracy: {}%".format(acc))

        endTime = time.time()
        elapsedTime = endTime - startTime
        print("Elapsed time: {} s".format(round(elapsedTime, 4)))

    if PERFORM_RAND_FOREST:

        startTime = time.time()

        acc = cm.performDecisionTreeClassifier(training, test)
        print("Rand. forest accuracy: {}%".format(acc))

        endTime = time.time()
        elapsedTime = endTime - startTime
        #print("Elapsed time: {}".format(timeAsStr))

    if PERFORM_CROSS_SVM:

        print("[5] Method chosen: SVM WITH CROSSVALIDATION")
        startTime = time.time()
        acc = cm.performCrossvalidationSVM(mat)
        endTime = time.time()

        elapsedTime = endTime - startTime
        print("Elapsed time: {} s".format(round(elapsedTime, 4)))

    if PERFORM_CROSS_KNN:

        print("[5] Method chosen: KNN WITH CROSSVALIDATION")
        startTime = time.time()
        acc = cm.performCrossvalidationKNN(mat, 7)
        endTime = time.time()

        elapsedTime = endTime - startTime
        print("Elapsed time: {} s".format(round(elapsedTime, 4)))

   
if __name__ == "__main__":
    main(sys.argv[1:])


def __init__(self):
    # store the number of points and radius
    self.numPoints = 24
