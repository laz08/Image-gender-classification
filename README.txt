- AUTHORS
		* Laura Cebollero
		* Carles Garriga
		* Balbina Virgili


- IMPLEMENTATION
	All of our code has been done in Python except the images extraction from subfolders, which have been done with Bash.

- PROJECT STRUCTURE
	.
	├── datasets
	│   ├── extractPictures.sh
	│   ├── facesInTheWild
	|	|	└── [...]
	│   ├── female_names.txt
	│   └── male_names.txt
	├── README.txt
	├── refs
	│   └── interesting_links.txt
	└── src
	    ├── classifiers
	    │   ├── ClassifierManager.py
	    │   ├── classifiers
	    │   │   ├── KNearestNeighbors.py
	    │   │   ├── MLP.py
	    │   │   └── SVM.py
	    │   ├── cnn_tensorflow.py
	    │   ├── Constants.py
	    │   ├── main.py
	    │   ├── utils_plot.py
	    │   └── Utils.py
	    └── preprocessing
	        ├── extractPictures.sh
	        ├── labels_merger.py
	        ├── merged_labels.txt
	        └── Utils.py

In this compressed file we can find three folders:

	- datasets: which is a folder with the labels of females, males and an empty folder call faceInTheWild where the images should be decompressed.
	- refs: which contains a txt inside with interesting URLs that we found interesting and relevant to our project.
	- src: which contains all the code.
		In this folder we have 2 subfolders: classifiers, which contains all the models we have used, and preprocessing, which has the scripts
		for the preprocessing of data.

	Please note that to facilitate the experiments on the LBP, feature extraction has been done on classifiers/main.py and not in the folder preprocessing,
	so that we could test different values for the number of points and the radius easily.

		- classifiers: contains the classifiers used on this project.

			To execute either KNN or SVM, one can do so by:
			$ python3.6 main.py -i ../preprocessing/merged_labels.txt

			* To choose which models will be computed (Knn or SVM) you have to put to True the flags
			that control which execution will be done:

			PERFORM_KNN = False
			PERFORM_SVM = False
			PERFORM_MLP = False
			PERFORM_CROSS_KNN = False
			PERFORM_CROSS_SVM = False

		- To execute the CNN model, the requirements are to have Tensorflow and Keras installed in your machine.
			Then, one can execute the CNN by executing cnn_tensorflow.py:
		
			$ python3.6 cnn_tensorflow.py



** WARNING **

We HAVE NOT included all the images for they weight 116MB. However, one can find all the images in here:
	http://vis-www.cs.umass.edu/lfw/

	More specifically, to download the funneled faces we used you can use the following link:
	http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz

Once these images have downloaded, you should uncompress them in the subfolders ./datasets/facesInTheWild
Then, you can change the extractPictures.sh inside the datasets folder and put your own path. Then you just run it and it will extract the images and
you are ready to go.
