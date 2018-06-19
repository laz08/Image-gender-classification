#!/usr/bin/python
# -*- coding: UTF-8 -*-

from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import Constants as const
import classifiers.KNearestNeighbors as Knn
import classifiers.SVM as SVM
import classifiers.MLP as MLP
from sklearn.metrics import log_loss
from math import exp


##--------------------------------------------------------------
##------------------------- CLASSIFIERS ------------------------
##--------------------------------------------------------------

def computeAccuracy(realData, predictions):
	femalePredCtr = 0
	malePredCtr = 0
	if(const._DEBUG):
		print("============")
	okCtr = 0
	failCtr = 0

	if(const._DEBUG):
		print(predictions)
	numPred = len(predictions)
	numReal = len(realData)
	if(const._DEBUG):
		print("Length " + str(numPred) + " - " + str(numReal))

	realLabels = [item[1] for item in realData]
	for i, predictedLabel in enumerate(predictions):
		if(const._DEBUG):
			print("Real:" + realLabels[i] + "Predicted: " + predictedLabel)
		if(str(predictedLabel).strip() == str(const._LABEL_MALE).strip()):
			malePredCtr += 1
		else:
			femalePredCtr += 1
		if(str(realLabels[i]).strip() == str(predictedLabel).strip()):
			okCtr += 1
		else:
			failCtr += 1

	print("    ==== RESULTS ====")
	print("    [*] OK {}".format(okCtr))
	print("    [*] Fail {}".format(failCtr))
	print("    [*] Male predicted {}".format(malePredCtr))
	print("    [*] Female predicted {}".format(femalePredCtr))
	return okCtr*100/len(realData)

def checkResultsPredicted(test, training, prediction, prediction_prob = None):

	if(const._DEBUG):
		print(prediction)
	numPred = len(prediction)
	numPredProb = len(prediction_prob)
	numReal = len(test)
	numTrain = len(training)
	if(const._DEBUG):
		print("Length " + str(numPred) + " - " + str(prediction_prob)+ " - " + str(numReal) + " - " + str(numTrain))

	acc = computeAccuracy(test, prediction)
	if(const._DEBUG):
		print("Type " + str(type(test)))
		print("Type " + str(type(prediction)))

	realLabels = [item[1] for item in test]
	tn, fp, fn, tp = confusion_matrix(realLabels, prediction).ravel()
	accuracy = (tp+tn)/len(prediction)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	
	print("\n    ==== METRICS ====")
	print("    [*] Accuracy: {}".format(round(accuracy, 4)))
	print("    [*] Precision: {}".format(round(precision, 4)))
	print("    [*] Recall: {}".format(round(recall, 4)))

	if (prediction_prob is not None):
		loss = log_loss([item[1] for item in test], prediction_prob)
		prob = exp(-loss)
		print("    [*] Log loss: {}".format(round(loss, 4)))
		print("    [*] Total prob: {}\n".format(round(prob, 4)))
	else:
		print("\n")
	#fpr, tpr, thresholds = roc_curve(realLabels, prediction, pos_label=2)
	#metrics.auc(fpr, tpr)
	return acc

def checkResultsCrossvalidation(scores,  acc = None, recall = None, prec = None):
	if(const._DEBUG):
		print(scores.keys())
		print(scores)

	test_neg_log_loss = scores['test_neg_log_loss'].mean()
	test_neg_log_loss_std = scores['test_neg_log_loss'].std()
	test_accuracy = scores['test_accuracy'].mean()
	train_neg_log_loss = scores['train_neg_log_loss'].mean()
	train_neg_log_loss_std = scores['train_neg_log_loss'].std()
	train_accuracy = scores['train_accuracy'].mean()
	fit_time = scores['fit_time'].mean()
	fit_time_std = scores['fit_time'].std()
	score_time = scores['score_time'].mean()
	score_time_std = scores['score_time'].std()

	print("\n    ==== METRICS ====")
	print("    [*] Fit time: %0.2f (+/- %0.2f) " % (fit_time, fit_time_std / 2))
	print("    [*] Training - Neg log loss: %0.2f (+/- %0.2f) " % (train_neg_log_loss, train_neg_log_loss_std / 2))
	print("    [*] Training - Accuracy: %0.2f" % (train_accuracy))
	print("    [*] Score time: %0.2f (+/- %0.2f) " % (score_time, score_time_std / 2))
	print("    [*] Test - Neg log loss: %0.2f (+/- %0.2f) " % (test_neg_log_loss, test_neg_log_loss_std / 2))
	print("    [*] Test - Accuracy: %0.2f \n" % (test_accuracy))

	#mean_loss = scores['test_loss'].mean()
	#std_loss = scores['test_loss'].std()

	#print("\n    ==== METRICS ====")
	#print("    [*] Mean Accuracy: {}".format(round(scores['test_acc'].mean(), 4)))
	#print("    [*] Mean Precision: {}".format(round(scores['test_rec'], 4)))
	#print("    [*] Mean Recall: {}".format(round(scores['test_prec'], 4)))
	#print("    [*] Mean Neg Log Loss: %0.2f (+/- %0.2f) \n" % (mean_loss, std_loss / 2))
	return test_accuracy



def performLinearSVC(training, test):

	prediction, prediction_prob = SVM.performSVM(training, test)
	#cross_val_score(clf, X, y, scoring='neg_log_loss')
	return checkResultsPredicted(test, training, prediction, prediction_prob)


def performKNeighbors(training, test, k):

	print("Selected K: {}".format(k))
	prediction, prediction_prob = Knn.performKNN(training, test, k)
	return checkResultsPredicted(test, training, prediction, prediction_prob)


def performMLPClassifier(training, test):
	
	prediction = MLP.performMLPClassifier(training, test)
	return checkResultsPredicted(test, training, prediction)


def performDecisionTreeClassifier(training, test):
   
	model = DecisionTreeClassifier()
	model.fit([item[2] for item in training], [item[1] for item in training])

	prediction = model.predict([item[2] for item in test])

	return checkResultsPredicted(test, training, prediction)


def performCrossvalidationSVM(mat):

	scores = SVM.performCrossValidationSVM(mat)

	return checkResultsCrossvalidation(scores)


def performCrossvalidationKNN(mat, k):

	scores, acc, recall, prec = Knn.performCrossValidationKNN(mat, k)

	return checkResultsCrossvalidation(scores, acc, recall, prec)




