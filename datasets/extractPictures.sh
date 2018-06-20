#!/bin/bash

PATH_PHOTOS="./lfw-deepfunneled"
PATH_DEST="./facesInTheWild"

find $PATH_PHOTOS > /tmp/files
while read var2
do
	tipus=`file "$var2" | awk -F ':' '{print $2}' | awk '{print $1}'`
	echo $tipus
	if [ "$tipus" = "directory" ]
	then
		nom=`echo "$var2" | awk -F '/' '{print $NF}'`
		echo $nom
		#cd "$PATH_PHOTOS/$nom"
		cp $PATH_PHOTOS/$nom/* $PATH_DEST
	fi
done < /tmp/files
echo "Finished extraction"
rm /tmp/files
