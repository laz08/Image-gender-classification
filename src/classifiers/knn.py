#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys, getopt, os
import time
import math, cv2
from matplotlib import pyplot as plot
import Utils

_PATH_TO_PHOTOS = "../../datasets/facesInTheWild/"
_NO_PHOTO_FLAG = "NO_PHOTO"
_LABEL_MALE = "MALE"
_LABEL_FEMALE = "FEMALE"


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

def computeEuclideanDistance(ind1, ind2, attrLength = 0, fromAttr=2, toAttr=4):
    # No attr length specified
    if(attrLength == 0):
        attrLength = len(ind1)

    d = 0
    # for x in range(fromAttr, toAttr):
    for x in range(attrLength):
        d += pow((ind1[x] - ind2[x]), 2)
    return (math.sqrt(d))

    # ind[2]: image properties
    # ind[3]: img histogram in RGB
def computeNeighbors(data, individualFeats, k = 5):
    neighborsDistances = []
    for n, neighbor in enumerate(data):
        if(neighbor[2] == _NO_PHOTO_FLAG): pass
        # We save the:
        # [0] neighbor index, [1] label and [2] distance
        dist = [n, neighbor[1], computeEuclideanDistance(data[n][2], individualFeats)]
        neighborsDistances.append(dist)

    # Order by least distance
    neighborsDistances.sort(key=operator.itemgetter(2))
    neighbors = []
    for i in range(k):
        neighbors.append(neighborsDistances[k])
    return neighbors

def getMostSimilar(neighbors):
    femaleCount = 0
    maleCount = 0
    for n in neighbors:
        if(n[1] == _LABEL_MALE):
            maleCount += 1
        else:
            femaleCount += 1

    if(maleCount > femaleCount):
        return _LABEL_MALE
    return _LABEL_FEMALE

def computeAccuracy(realData, predictions):
    okCtr = 0
    failCtr = 0
    for i, rd in enumerate(realData):
        for p in predictions:
            # If index of real/test data is found
            if(i == p[0]):
                if(rd[1] == p[1]):
                    okCtr += 1
                else:
                    failCtr += 1
                break

    return okCtr*100/len(realData)

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

def usage():

    print ('Usage: main.py -i <inputfile>')
    sys.exit(2)

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
    idxRead = 0         # Idx to be read later of images read
    propToRead = 10     # Proportion of images to be read
    startTime = time.time()
    for i, ind in enumerate(mat):
        image_path = os.path.join(_PATH_TO_PHOTOS, ind[0])
        if os.path.exists(_PATH_TO_PHOTOS):
            img = cv2.imread(image_path, 0)
            if(img is None):
                print("Could not read image")
                ind.append(_NO_PHOTO_FLAG)
            else:
                #showImage(img)
                ind.append(getImageFeatures(img))
                ind.append(getImageHistogram(img))
                
            if(i > 0 and i % 1000 == 0):
                print("[IMG] processed {}/{}. {}%".format(i, len(mat), round(i*100/len(mat), 2)))
                if(i*100/len(mat) > propToRead):
                    idxRead = i
                    break

        else:
            ind.append(_NO_PHOTO_FLAG)
            print("Path does not exist" + str(image_path))

    endTime = time.time()
    elapsedTime = endTime - startTime
    timeAsStr = time.strftime("%M:%S", time.gmtime(elapsedTime))
    print("Elapsed time: {}".format(timeAsStr))

    mat = mat[0:idxRead]
    training, test = splitTrainingTestSet(mat)

    # ind[0]: image path
    # ind[1]: Label 
    # ind[2]: image properties
    # ind[3]: img histogram in RGB
    predictions = []
    for i, ind in enumerate(test):
        if(ind[2] == _NO_PHOTO_FLAG):
            pass
        neighbors = computeNeighbors(training, ind, 1)
        result = getMostSimilar(neighbors)
        predictions.append([i, result])

    acc = computeAccuracy(test, predictions)
    print("Accuracy: {}%".format(acc))

    # print(computeEuclideanDistance(mat[0][2], mat[3][2]))

if __name__ == "__main__":
   main(sys.argv[1:])
