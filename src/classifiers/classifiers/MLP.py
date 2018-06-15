
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt



def performMLPClassifier(training, test):

    model = MLPClassifier(solver='lbfgs')
    model.fit([item[2] for item in training], [item[1] for item in training])

    prediction = model.predict([item[2] for item in test])

    return prediction
