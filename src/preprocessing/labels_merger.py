#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys, getopt
import os

_NAMES_PATH = "../../datasets"
_MALE_LABELS = "/male_names.txt"
_FEMALE_LABELS = "/female_names.txt"
_MERGED_FILE = "merged_labels.txt"


def writeToFile(fileOut, contents):
    f = open(fileOut, 'w+')
    f.write(str(contents))
    f.close()
    print ("Finished writing data")


def readLineAsArrayWithAppend(filein, toAppend):
    print(filein)
    array = []
    with open(filein, "r") as ins:
        for line in ins:
            name = line
            if (line != "\n"):
                array.append(line.replace("\n", "; " + toAppend))

    return array


def extractNames(rawLabels):
    cleanLabels = []
    for lab in rawLabels:
        comps = lab.split("_")
        name = comps[0]
        if len(comps) > 1:
            name = name + "_" + comps[1]
        cleanLabels.append(name)

    return cleanLabels


def main():
    # Extract names with lines
    femaleNames = readLineAsArrayWithAppend(_NAMES_PATH + _FEMALE_LABELS, "FEMALE")
    maleNames = readLineAsArrayWithAppend(_NAMES_PATH + _MALE_LABELS, "MALE")

    finalStr = ""

    print("Female names size: {}".format(len(femaleNames)))
    print("Male names size: {}".format(len(maleNames)))

    subsetMaleNames = maleNames[0:len(femaleNames)]
    print("Subset male names size: {}".format(len(subsetMaleNames)))

    useZip = False
    if useZip:
        names = list(zip(femaleNames, subsetMaleNames))

        for name in names:
            finalStr = finalStr + name[0] + "\n"
            finalStr = finalStr + name[1] + "\n"

    else:
        names = list((femaleNames + subsetMaleNames))
        for name in names:
            finalStr = finalStr + name + "\n"

    writeToFile(_MERGED_FILE, finalStr)

    print ("Finished creating labels.")


if __name__ == "__main__":
    main()
