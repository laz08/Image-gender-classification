
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestNeighbors

def performKNN(training, test):

    # train and evaluate a k-NN classifer on the histogram
    # representations

    model = KNeighborsClassifier(10)
    model.fit([item[2] for item in training], [item[1] for item in training])

    #acc = model.score([item[2] for item in test], [item[1] for item in test])
    prediction = model.predict([item[2] for item in test])

    return prediction
    