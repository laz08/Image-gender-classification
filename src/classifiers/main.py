#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import getopt
import time

import Utils
import ClassifierManager as cm

PERFORM_KNN = False
PERFORM_SVM = True
PERFORM_MLP = False
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

        k = 7    
        cm.performKNeighbors(training, test, k)

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


    if PERFORM_CROSS_SVM:

        print("[5] Method chosen: SVM WITH CROSSVALIDATION")
        startTime = time.time()
        
        accuracyArray = []
        accuracyStdArray = []
        negLossArray = []
        
        diffC = [0.1, 0.5, 1, 10, 50, 100, 200, 500, 1000]
        for c in diffC:
            acc, accStd, negLoss, negLossStd  = cm.performCrossvalidationSVM(mat, c)
            accuracyArray.append(acc)
            accuracyStdArray.append(accStd)
            negLossArray.append(negLoss)

        fileTime = time.time()
        fileName = "SVM_Results_" + str(fileTime) + ".csv"
        Utils.appendLineToFile(fileName, "C ; Accuracy ; Accuracy Std; Log Loss;")
        for c, val in enumerate(diffC):
            Utils.appendLineToFile(fileName, 
                str(val) + "; " + \
                str(round(accuracyArray[c], 4)) + "; " + \
                str(round(accuracyStdArray[c], 4)) + "; " + \
                str(round(negLossArray[c], 4)) + "; "
                )

        endTime = time.time()

        elapsedTime = endTime - startTime
        print("Elapsed time: {} s".format(round(elapsedTime, 4)))

    if PERFORM_CROSS_KNN:

        print("[5] Method chosen: KNN WITH CROSSVALIDATION")
        startTime = time.time()
        
        accuracyArray = []
        accuracyStdArray = []
        negLossArray = []
        negLossStdArray = []

        diffK = [3, 5, 7, 13, 21, 51, 101]
        for k in diffK:
            acc, accStd, negLoss, negLossStd = cm.performCrossvalidationKNN(mat, k)
            accuracyArray.append(acc)
            accuracyStdArray.append(accStd)
            negLossArray.append(negLoss)
            negLossStdArray.append(negLossStdArray)
    
        fileTime = time.time()
        fileName = "Knn_Results_" + str(fileTime) + ".csv"
        Utils.appendLineToFile(fileName, "K ; Accuracy ; Accuracy Std; Log Loss;")
        for k, val in enumerate(diffK):
            Utils.appendLineToFile(fileName, 
                str(val) + "; " + \
                str(round(accuracyArray[k], 4)) + "; " + \
                str(round(accuracyStdArray[k], 4)) + "; " + \
                str(round(negLossArray[k], 4)) + "; "
                )
                
                
        endTime = time.time()
        elapsedTime = endTime - startTime
        print("Elapsed time: {} s".format(round(elapsedTime, 4)))

   
if __name__ == "__main__":
    main(sys.argv[1:])


def __init__(self):
    # store the number of points and radius
    self.numPoints = 24
