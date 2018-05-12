#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys, getopt
import math

import Utils

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


def extractImageProperties(ind):
    img = ind[0]
    
    return


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
    print(mat[0][0])

    training, test = splitTrainingTestSet(mat, 0.8)
    print (len(training))
    print (len(test))
    for ind in training:
        extractImageProperties(ind)



if __name__ == "__main__":
   main(sys.argv[1:])
