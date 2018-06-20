#!/bin/bash

PATH_PHOTOS="/home/nora/git/machine-learning-project/datasets/facesRAW/lfw-deepfunneled"
PATH_DEST="/home/nora/git/machine-learning-project/datasets/facesInTheWild"

find $PATH_PHOTOS > /tmp/files
while read var2
do
	tipus=`file "$var2" | awk -F ':' '{print $2}' | awk '{print $1}'`
	echo $tipus
	if [ "$tipus" = "directory" ]
	then
		nom=`echo "$var2" | awk -F '/' '{print $NF}'`
		echo $nom
		cp $PATH_PHOTOS/$nom/* $PATH_DEST
	fi
done < /tmp/files
echo "Extraction finished."
rm /tmp/files
