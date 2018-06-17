
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestNeighbors

def performKNN(training, test, k):

    # train and evaluate a k-NN classifer on the histogram
    # representations

    model = KNeighborsClassifier(k)
    model.fit([item[2] for item in training], [item[1] for item in training])

    #acc = model.score([item[2] for item in test], [item[1] for item in test])

    prediction_prob = model.predict_proba([item[2] for item in test])
    #print("prediction " + str(prediction_prob))
    prediction = model.predict([item[2] for item in test])

    return prediction, prediction_prob
    