#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys, getopt
import Utils as utils
import os

_NAMES_PATH =  "../../datasets"
_MALE_LABELS = "/male_names.txt"
_FEMALE_LABELS = "/female_names.txt"
_NEW_LABELS_PATH = "photo_path.txt"

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

    photosDir = changeToRelativePath(_PHOTOS_DIR)
    if(isMale):
        LABELS = _MALE_LABELS
    else:
        LABELS = _FEMALE_LABELS

    paths = []
    for name in setOfNames:
        break
        pathsToFile = getPathsToFile(name, photosDir)
        paths.append(pathsToFile)
        #for p in pathsToFile:
        #    utils.appendLineToFile(LABELS, p)
        print(paths[0])
        break


def main():

    femaleNames = utils.readLineAsArray(_NAMES_PATH + _FEMALE_LABELS)
    #maleNames = utils.readLineAsArray(_MALE_NAMES_PATH)

    print(femaleNames[0])
    if(False):
        utils.writeToFile(_MALE_LABELS, "")
        utils.writeToFile(_FEMALE_LABELS, "")

    femaleNames = extractNames(femaleNames)
    #maleNames = extractNames(maleNames)

    #createLabels(femaleNames, False)
    #createLabels(maleNames, True)

    print ("Finished creating labels.")





if __name__ == "__main__":
   main()
