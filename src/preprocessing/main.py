#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys, getopt
import Utils as utils
import os

_FEMALE_NAMES_PATH =  "../../datasets/faces_in_the_wild/female_names.txt"
#_FEMALE_NAMES_FILE =  "female_names.txt"

_FEMALE_NAMES_PATH1 =  "../../datasets/faces_in_the_wild/female_names.txt"
_MALE_NAMES_PATH =  "../../datasets/faces_in_the_wild/male_names.txt"

_PHOTOS_DIR = "../../datasets/faces_in_the_wild/lfw-deepfunneled/"

_MALE_LABELS = "malePaths.txt"
_FEMALE_LABELS = "femalePaths.txt"

def getRelativePath(relativePath):
    #os.chdir(os.path.normpath(os.path.join(os.getcwd(), relativePath)))
    return os.path.normpath(os.path.join(os.getcwd(), relativePath))

def extractNames(rawLabels):
    cleanLabels = []
    for lab in rawLabels:
        comps = lab.split("_")
        cleanLabels.append(comps[0] + "_" + comps[1])

    return cleanLabels

def getPathsToFile(name, parentPath):
    
    paths = []
    dirs = [d for d in os.listdir(parentPath)]
    for d in dirs:
        if(d == name):
            destdir = parentPath + d
            files = [f for f in os.listdir(destdir) if os.path.isfile(os.path.join(destdir, f))]
            for f in files:
                paths.append(destdir + "/" + f)

        break
    return paths
    
# Creates label for a set of names.
# isMale = True if male, false otherwise.
def createLabels(setOfNames, isMale):

    photosDir = getRelativePath(_PHOTOS_DIR)
    if(isMale):
        LABELS = _MALE_LABELS
    else:
        LABELS = _FEMALE_LABELS

    paths = []
    for name in setOfNames:
        pathsToFile = getPathsToFile(name, photosDir)
        paths.append(pathsToFile)
        #for p in pathsToFile:
        #    utils.appendLineToFile(LABELS, p)
        print(paths[0])
        break


def main():

    path = getRelativePath(_FEMALE_NAMES_PATH)

    femaleNames = utils.readLineAsArray(path)
    #maleNames = utils.readLineAsArray(_MALE_NAMES_PATH)

    print(femaleNames[0])
    if(False):
        utils.writeToFile(_MALE_LABELS, "")
        utils.writeToFile(_FEMALE_LABELS, "")

    #print(os.getcwd())
    photosPath = getRelativePath(_PHOTOS_DIR)

    femaleNames = extractNames(femaleNames)
    
    #maleNames = extractNames(maleNames)

    createLabels(femaleNames, False)
    #createLabels(maleNames, True)

    print ("Finished creating labels.")





if __name__ == "__main__":
   main()
