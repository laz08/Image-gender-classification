#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys, getopt, os
import time
import math, cv2
from matplotlib import pyplot as plot
import Utils

_PATH_TO_PHOTOS = "../../datasets/facesInTheWild/"

def getImageFeatures(img, size = (150, 150)):
    "Resizes to 150x150 and Flatten: List of raw pixel intensities"
    return cv2.resize(img, size).flatten()


def plotHistogram(hist):
    "Plots histogram for debugging"
    plot.figure()
    plot.title("RGB histogram for image")
    plot.xlabel("Color bins")
    plot.ylabel("# pixels")
    plot.plot(hist)
    plot.xlim([0, 255])
    plot.show()

def getImageHistogram(img):
    "Returns color histogram"
    rgb_channel = ['b', 'g', 'r']
    grayscale = [0]
    # cv2.calcHist([images], channels, mask, histSize, ranges)
    hist = cv2.calcHist(
        [img],          # Images list
         grayscale,    # Channel. We want RGB
         None,          # Mask. We dont want to specify a region... yet
         [256],      # Bins. 256 / 8
         [0, 256])      # Range of colors
    cv2.normalize(hist, hist)
    # plotHistogram(hist)
    # Return hist as vect
    return hist.flatten()

def computeEuclideanDistance(ind1, ind2, attrLength = 0):
    # No attr length specified
    if(attrLength == 0):
        attrLength = len(instance1)

    d = 0
    for x in range(attrLength):
        d += pow((ind1[x] - ind2[x]), 2)
    return (math.sqrt(d))

def splitTrainingTestSet(data, trainingProp = 0.8):
    # todo make it randomly, not this straightforward
    idxToCut = int(trainingProp * len(data))
    training = data[0:idxToCut]
    test = data[idxToCut+1:len(data)]
    return training, test

def showImage(img):
    cv2.imshow('',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def main(argv):

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

    # print (len(training))
    # print (len(test))

    startTime = time.time()
    for i, ind in enumerate(mat):
        image_path = os.path.join(_PATH_TO_PHOTOS, ind[0])
        if os.path.exists(_PATH_TO_PHOTOS):
            img = cv2.imread(image_path, 0)
            if(img is None):
                print("Could not read image")
            else:
                #showImage(img)
                ind.append(getImageFeatures(img))
                ind.append(getImageHistogram(img))
                
            if(i > 0 and i % 1000 == 0):
                print("[IMG] % processed {}/{}".format(i, len(mat)))

        else:
            print("Path does not exist" + str(image_path))

    endTime = time.time()
    elapsedTime = endTime - startTime
    timeAsStr = time.strftime("%M:%S", time.gmtime(elapsedTime))
    print("Elapsed time: {}".format(timeAsStr))
    training, test = splitTrainingTestSet(mat)

    # ind[0]: image path
    # ind[1]: Label 
    # ind[2]: image properties
    # ind[3]: img histogram in RGB

if __name__ == "__main__":
   main(sys.argv[1:])
