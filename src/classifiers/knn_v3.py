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
import numpy as np


_PATH_TO_PHOTOS = "../../datasets/facesInTheWild/"
_LABEL_MALE = "MALE"
_LABEL_FEMALE = "FEMALE"


def getImageFeatures(img, size = (150, 150)):
    "Resizes to 150x150 and Flatten: List of raw pixel intensities"
    #cv2.face.createLBPHFaceRecognizer()
    model = cv2.face.FisherFaceRecognizer_create()
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
        d += pow((float(ind1[x]) - float(ind2[x])), 2)
    return (math.sqrt(d))

    # ind[2]: image properties
    # ind[3]: img histogram in RGB
def computeNeighbors(data, individual, k = 5):
    neighborsDistances = []
    for n, neighbor in enumerate(data):
        # We save the:
        # [0] neighbor index, [1] label and [2] distance
        dist = [n, neighbor[1], computeEuclideanDistance(data[n][2], individual[2])]
        neighborsDistances.append(dist)

    # Order by least distance
    neighborsDistances.sort(key=operator.itemgetter(2))
    neighbors = []
    for i in range(k):
        print(neighborsDistances[i])
        neighbors.append(neighborsDistances[i])
    return neighbors

def getMostSimilar(neighbors):
    femaleCount = 0
    maleCount = 0
    for n in neighbors:
        if(str(n[1]).strip() == str(_LABEL_MALE).strip()):
            maleCount += 1
        else:
            femaleCount += 1

    if(maleCount > femaleCount):
        return _LABEL_MALE
    return _LABEL_FEMALE

def computeAccuracy(realData, predictions):
    femalePredCtr = 0
    malePredCtr = 0
    print("============")
    okCtr = 0
    failCtr = 0
    for i, rd in enumerate(realData):
        # print("Real data:{}".format(rd[1]))
        for p in predictions:
            # If index of real/test data is found
            if(i == p[0]):

                if(str(p[1]).strip() == str(_LABEL_MALE).strip()):
                    malePredCtr += 1
                else:
                    femalePredCtr += 1

               # print("Predicted: {} || Real: {}".format(p[1], rd[1]))
               # print(rd[1])
               # print(p[1])
               # print(str(rd[1]).strip() == str(p[1]).strip())
                if(str(rd[1]).strip() == str(p[1]).strip()):
                    okCtr += 1
                else:
                    failCtr += 1
                break

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

def showImage(img):
    cv2.imshow('',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def usage():

    print ('Usage: main.py -i <inputfile>')
    sys.exit(2)

def describe(numPoints, image, radius, eps=1e-7):
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = feature.local_binary_pattern(image, numPoints,
        radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
        bins=np.arange(0, numPoints + 3),
        range=(0, numPoints + 2))

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    #print(hist)
    # return the histogram of Local Binary Patterns
    return hist


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
    toHave = 1000
    for i in mat:
        if(str(i[1]).strip() == str(_LABEL_MALE) and maleCtr < toHave):
            maleCtr +=1
            matTmp.append(i)
        elif(str(i[1]).strip() == str(_LABEL_FEMALE) and femaleCtr < toHave):
            femaleCtr +=1
            matTmp.append(i)
        if(maleCtr >= toHave and femaleCtr >= toHave):
            break

    mat = matTmp
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
                #print("Image read")
                resized = cv2.resize(image, (128, 128), cv2.INTER_LINEAR)
                #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                hist = describe(24, resized, 8)
                data.append(hist)
                ind.append(hist)
                print("label" + str(ind[1]))
                labels.append(ind[1])
        else:
            imgNotRead.append(i)
            print("Path does not exist" + str(image_path))

    # train a Linear SVM on the data
    model = LinearSVC(C=100.0, random_state=42)
    model.fit(data, labels)

    image = cv2.imread('../../datasets/facesInTheWild/Adam_Sandler_0002.jpg', 0)
    resized = cv2.resize(image, (150, 150), cv2.INTER_LINEAR)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = describe(24, resized, 8)
    prediction = model.predict(hist.reshape(1,-1))[0]
    print("Prediction" + str(prediction))


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