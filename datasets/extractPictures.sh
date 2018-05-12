#!/bin/bash
#copiar fitxers mp3 del usb (muntat correctament)
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
		#cd "$PATH_PHOTOS/$nom"
		cp $PATH_PHOTOS/$nom/* $PATH_DEST
	fi
done < /tmp/files
echo "Fi de la copia"
rm /tmp/files
