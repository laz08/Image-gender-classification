#!/usr/bin/python
# -*- coding: UTF-8 -*-

from skimage import feature
import sys, getopt, os
import time
import random
import math
import cv2
import numpy as np

import Constants as const

def writeToFile(fileOut, contents):
    f = open(fileOut, 'w+')
    f.write(str(contents))
    f.close()
    print ("Finished generating data")


def readFromFile(filein):
    array = []
    with open(filein, "r") as ins:
        for line in ins:
            array.append(line)

    return array

    

def readFromFileAsArray(filein):
    file = open(filein, 'r')
    contents = file.read()
    file.close()
    return contents


def readAsMatrix(filein):
    print(filein)
    mat = []
    with open(filein, "r") as ins:
        for line in ins:
            comps = line.split("; ")
            if(len(comps) > 1):
                mat.append([comps[0], comps[1]])

    return mat

def appendLineToFile(fileout, line):
    with open(fileout, "a") as f:
        f.write(line + "\n")



def splitTrainingTestSet(data, trainingProp = 0.8, rand = True):
    training = []
    test = []
    
    if(rand):
        for d in data:
            r=random.random()
            if(r <= trainingProp):
                training.append(d)
            else:
                test.append(d)
    else:
        idxToCut = int(trainingProp * len(data))
        training = data[0:idxToCut]
        test = data[idxToCut+1:len(data)]
    
    return training, test


# --------------------------------------
# ----------------- IMAGES -------------
# --------------------------------------
def getImageFeatures(img, size =  (150, 150)):
    "Resizes to 150x150 and Flatten: List of raw pixel intensities"
    #cv2.face.createLBPHFaceRecognizer()
    model = cv2.face.FisherFaceRecognizer_create()
    return cv2.resize(img, size).flatten()



def describe(numPoints, image, radius, eps=1e-7):
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = feature.local_binary_pattern(image, numPoints, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
        bins=np.arange(0, numPoints + 3),
        range=(0, numPoints + 2))

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    #print(hist)
    # return the histogram of Local Binary Patterns
    return hist


def cropToFace(img):
    # Y axis Crop
    # y = 64
    y = 80
    # h = 116
    h = 100


    # X axis crop
    x = 80
    #x = 90
    w = 90
    #w = 80

    face = img[y:y+h, x:x+w]

    return face



def plotImage(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def forceGenderParityUpToN(mat, N):

    # ADD 50 50 %
    matTmp = []
    maleCtr = 0
    femaleCtr = 0
    toHave = N
    for i in mat:
        if(str(i[1]).strip() == str(const._LABEL_MALE) and maleCtr < toHave):
            maleCtr +=1
            matTmp.append(i)
        elif(str(i[1]).strip() == str(const._LABEL_FEMALE) and femaleCtr < toHave):
            femaleCtr +=1
            matTmp.append(i)
        if(maleCtr >= toHave and femaleCtr >= toHave):
            break

    return matTmp

def readImages(mat):

    numPoints = 24
    rad = 3
    print("[*] Extracting features using LBP, {} points, radius {}".format(numPoints, rad))

    imgNotRead = []
    # print (len(training))
    # print (len(test))
    idxRead = len(mat)         # Idx to be read later of images read
    propToRead = 100     # Proportion of images to be read
    startTime = time.time()
    for i, ind in enumerate(mat):
        image_path = os.path.join(const._PATH_TO_PHOTOS, ind[0])
        if(const._DEBUG):
            print("Image path")
            print(image_path)
        if os.path.exists(const._PATH_TO_PHOTOS):
            image = cv2.imread(image_path, 0)
            if(image is None):

                imgNotRead.append(i)
                if(const._DEBUG):
                    print("Could not read image")
            
            else:
                image = cropToFace(image)               
                height, width = image.shape
                resized = cv2.resize(image, (width, height))

                #plotImage(resized)

                hist = describe(numPoints, resized, rad)
                cv2.normalize(hist, hist)
                hist.flatten()

                ind.append(hist)
               
        else:
            imgNotRead.append(i)
            if(const._DEBUG):
                print("Path does not exist" + str(image_path))

    if(const._DEBUG):                
        print("Removing images that could not be read...")
    mat = mat[0:idxRead]
    for imgIdx in imgNotRead:
        del mat[imgIdx]
    if(const._DEBUG):
        print("Done.")
    return mat