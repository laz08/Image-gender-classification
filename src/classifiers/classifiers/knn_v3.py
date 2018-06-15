#!/usr/bin/python
# -*- coding: UTF-8 -*-

from skimage import feature
import sys, getopt, os
import time
import math
import cv2
import operator, random
from matplotlib import pyplot as plot
import Utils
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


_PATH_TO_PHOTOS = "../../datasets/facesInTheWild/"
_LABEL_MALE = "MALE"
_LABEL_FEMALE = "FEMALE"


def computeAccuracy(realData, predictions):
    femalePredCtr = 0
    malePredCtr = 0
    print("============")
    okCtr = 0
    failCtr = 0

    print(predictions)
    numPred = len(predictions)
    numReal = len(realData)
    print("Length " + str(numPred) + " - " + str(numReal))

    realLabels = [item[1] for item in realData]
    for i, predictedLabel in enumerate(predictions):
        print("Real:" + realLabels[i] + "Predicted: " + predictedLabel)
        if(str(predictedLabel).strip() == str(_LABEL_MALE).strip()):
            malePredCtr += 1
        else:
            femalePredCtr += 1
        if(str(realLabels[i]).strip() == str(predictedLabel).strip()):
            okCtr += 1
        else:
            failCtr += 1

    print("OK {}".format(okCtr))
    print("Fail {}".format(failCtr))
    print("Male predicted {}".format(malePredCtr))
    print("Female predicted {}".format(femalePredCtr))
    return okCtr*100/len(realData)

def splitTrainingTestSet(data, trainingProp = 0.8):
    training = []
    test = []
    for d in data:
        r=random.random()
        if(r <= trainingProp):
            training.append(d)
        else:
            test.append(d)
    #idxToCut = int(trainingProp * len(data))
    #training = data[0:idxToCut]
    #test = data[idxToCut+1:len(data)]
    return training, test


def usage():

    print ('Usage: main.py -i <inputfile>')
    sys.exit(2)

def main(argv):

    data = []
    labels = []
    if len(argv) < 1:
         usage()
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
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
    
    # ADD 50 50 %
    matTmp = []
    maleCtr = 0
    femaleCtr = 0
    toHave = 20
    for i in mat:
        if(str(i[1]).strip() == str(_LABEL_MALE) and maleCtr < toHave):
            maleCtr +=1
            matTmp.append(i)
        elif(str(i[1]).strip() == str(_LABEL_FEMALE) and femaleCtr < toHave):
            femaleCtr +=1
            matTmp.append(i)
        if(maleCtr >= toHave and femaleCtr >= toHave):
            break

    #image_path = os.path.join(_PATH_TO_PHOTOS, mat[0][0])
    #image = cv2.imread(image_path)
    #Utils.plotImage(image)
    #image_path = os.path.join(_PATH_TO_PHOTOS, mat[1][0])
    #image = cv2.imread(image_path)
    #Utils.plotImage(image)
    #############################################

    imgNotRead = []
    # print (len(training))
    # print (len(test))
    idxRead = len(mat)         # Idx to be read later of images read
    propToRead = 100     # Proportion of images to be read
    startTime = time.time()
    for i, ind in enumerate(mat):
        image_path = os.path.join(_PATH_TO_PHOTOS, ind[0])
        if os.path.exists(_PATH_TO_PHOTOS):
            image = cv2.imread(image_path, 0)
            if(image is None):
                imgNotRead.append(i)
                print("Could not read image")
            else:
               image = cropToFace(image)
               
                height, width = image.shape
                resized = cv2.resize(image, (width, height))

                hist = Utils.describe(24, resized, 8)
                cv2.normalize(hist, hist)
                hist.flatten()
                data.append(hist)
                ind.append(hist)
                #print("label" + str(ind[1]))
                labels.append(ind[1])
        else:
            imgNotRead.append(i)
            print("Path does not exist" + str(image_path))

    print("Removing images that could not be read...")
    mat = mat[0:idxRead]
    for imgIdx in imgNotRead:
        del mat[imgIdx]
    print("Done.")

    training, test = splitTrainingTestSet(mat)

    # train a Linear SVM on the data
    acc = performLinearSVC(training, test, mat)
    # acc = performKNeighbors(training, test)
    #acc = performMLPClassifier(training, test)
    #acc = performDecisionTreeClassifier(training, test)

    print("Accuracy: {}%".format(acc))

    endTime = time.time()
    elapsedTime = endTime
    timeAsStr = time.strftime("%M:%S", time.gmtime(elapsedTime))
    print("Elapsed time: {}".format(timeAsStr))

    endTime = time.time()
    print("Done.")
    elapsedTime = endTime
    timeAsStr = time.strftime("%M:%S", time.gmtime(elapsedTime))
    print("Elapsed time: {}".format(timeAsStr))

    # print(computeEuclideanDistance(mat[0][2], mat[3][2]))

if __name__ == "__main__":
   main(sys.argv[1:])

def __init__(self):
    # store the number of points and radius
	self.numPoints = 24