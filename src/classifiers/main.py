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
    imagesNumberPerGender = 3000    # If the num. of images is greater than the maximum of them, then it reads all of them
    print("[2] Forcing gender parity up to {} images per gender".format(imagesNumberPerGender))
    mat = Utils.forceGenderParityUpToN(mat, imagesNumberPerGender)
    #############################################

    print("[3] Extracting features...")
    mat = Utils.readImages(mat)

    print("[4] Splitting on training/test set...")
    training, test = Utils.splitTrainingTestSet(mat, 0.8, False)

    if PERFORM_KNN:

        print("[5] Method chosen: KNN")
        startTime = time.time()

        acc = cm.performKNeighbors(training, test, 7)
        #print("Accuracy with Knn: {}%".format(acc))

        endTime = time.time()
        elapsedTime = endTime - startTime
        print("Elapsed time: {} s".format(round(elapsedTime, 4)))

    if PERFORM_SVM:

        print("[5] Method chosen: SVM")
        startTime = time.time()

        acc = cm.performLinearSVC(training, test, mat)
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
        print("Elapsed time: {}".format(timeAsStr))
    
   
if __name__ == "__main__":
    main(sys.argv[1:])


def __init__(self):
    # store the number of points and radius
    self.numPoints = 24
