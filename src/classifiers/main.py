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

PERFORM_KNN = PERFORM_KNN and not PERFORM_SVM and not PERFORM_MLP and not PERFORM_RAND_FOREST
PERFORM_SVM = PERFORM_SVM and not PERFORM_KNN and not PERFORM_MLP and not PERFORM_RAND_FOREST
PERFORM_MLP = PERFORM_MLP and not PERFORM_SVM and not PERFORM_KNN and not PERFORM_RAND_FOREST
PERFORM_RAND_FOREST = PERFORM_RAND_FOREST and not PERFORM_KNN and not PERFORM_SVM and not PERFORM_MLP


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

    print ("Reading from file " + filein)
    mat = Utils.readAsMatrix(filein)

    #############################################
    imagesNumberPerGender = 100		# If the num. of images is greater than the maximum of them, then it reads all of them
    mat = Utils.forceGenderParityUpToN(mat, imagesNumberPerGender)
    #############################################

    mat = Utils.readImages(mat)

    training, test = Utils.splitTrainingTestSet(mat)

    # train a Linear SVM on the data
    if PERFORM_KNN:

        startTime = time.time()

        acc = cm.performKNeighbors(training, test)
        print("Accuracy with Knn: {}%".format(acc))

        endTime = time.time()
        elapsedTime = endTime - startTime
        timeAsStr = time.strftime("%M:%S", time.gmtime(elapsedTime))
        print("Elapsed time: {}".format(timeAsStr))

    if PERFORM_SVM:

        startTime = time.time()

        acc = cm.performLinearSVC(training, test, mat)
        print("SVM accuracy: {}%".format(acc))

        endTime = time.time()
        elapsedTime = endTime - startTime
        timeAsStr = time.strftime("%M:%S", time.gmtime(elapsedTime))
        print("Elapsed time: {}".format(timeAsStr))

    if PERFORM_MLP:

        startTime = time.time()

        acc = cm.performMLPClassifier(training, test)
        print("MLP accuracy: {}%".format(acc))

        endTime = time.time()
        elapsedTime = endTime - startTime
        timeAsStr = time.strftime("%M:%S", time.gmtime(elapsedTime))
        print("Elapsed time: {}".format(timeAsStr))

    if PERFORM_RAND_FOREST:

        startTime = time.time()

        acc = cm.performDecisionTreeClassifier(training, test)
        print("Rand. forest accuracy: {}%".format(acc))

        endTime = time.time()
        elapsedTime = endTime - startTime
        timeAsStr = time.strftime("%M:%S", time.gmtime(elapsedTime))
        print("Elapsed time: {}".format(timeAsStr))
    
   
if __name__ == "__main__":
    main(sys.argv[1:])


def __init__(self):
    # store the number of points and radius
    self.numPoints = 24
