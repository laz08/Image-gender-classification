#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys, getopt
import Utils as utils
import os

_FEMALE_NAMES_PATH =  "../../datasets/faces_in_the_wild/"
_FEMALE_NAMES_FILE =  "female_names.txt"

_FEMALE_NAMES_PATH1 =  "../../datasets/faces_in_the_wild/female_names.txt"
_MALE_NAMES_PATH =  "../../datasets/faces_in_the_wild/male_names.txt"

_PHOTOS_DIR = "../../datasets/faces_in_the_wild/lfw-deepfunneled/"

_MALE_LABELS = "malePaths.txt"
_FEMALE_LABELS = "femalePaths.txt"

def changeToRelativePath(relativePath):
    os.chdir(os.path.normpath(os.path.join(os.getcwd(), relativePath)))

def extractNames(rawLabels):
    cleanLabels = []
    for lab in rawLabels:
        comps = lab.split("_")
        cleanLabels.append(comps[0] + "_" + comps[1])

    return cleanLabels

def getPathsToFile(name):
    
    paths = []
    dirs = [d for d in os.listdir(os.getcwd())]
    for d in dirs:
        if(d == name):
            destdir = os.getcwd() + d
            files = [f for f in os.listdir(destdir) if os.path.isfile(os.path.join(destdir, f))]
            for f in files:
                paths.append(destdir + "/" + f)

        break
    return paths
    
# Creates label for a set of names.
# isMale = True if male, false otherwise.
def createLabels(setOfNames, isMale):

    if(isMale):
        LABELS = _MALE_LABELS
    else:
        LABELS = _FEMALE_LABELS

    for name in setOfNames:
        pathsToFile = getPathsToFile(name)
        for p in pathsToFile:
            utils.appendLineToFile(LABELS, p)


def main():

    changeToRelativePath(_FEMALE_NAMES_PATH)

    femaleNames = utils.readLineAsArray(_FEMALE_NAMES_FILE)
    #maleNames = utils.readLineAsArray(_MALE_NAMES_PATH)

    if(False):
        utils.writeToFile(_MALE_LABELS, "")
        utils.writeToFile(_FEMALE_LABELS, "")

    changeToRelativePath(_PHOTOS_DIR)

    femaleNames = extractNames(femaleNames)
    #maleNames = extractNames(maleNames)

    createLabels(femaleNames, False)
    #createLabels(maleNames, True)

    print ("Finished creating labels.")





if __name__ == "__main__":
   main()
