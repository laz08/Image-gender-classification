from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from matplotlib import pyplot as plt


def performKNN(training, test, k):

    # train and evaluate a k-NN classifer on the histogram
    # representations

    model = KNeighborsClassifier(k)
    model.fit([item[2] for item in training], [item[1] for item in training])

    #acc = model.score([item[2] for item in test], [item[1] for item in test])

    prediction_prob = model.predict_proba([item[2] for item in test])
    #print("prediction " + str(prediction_prob))
    prediction = model.predict([item[2] for item in test])
    plt.scatter([item[1] for item in test], prediction)
    return prediction, prediction_prob

def performCrossValidationKNN(mat, k):

    model = KNeighborsClassifier(k)

    #pipeline = Pipeline([('transformer', scalar), ('estimator', clf)])

    cv = KFold(n_splits=10)

    print("[*] K value: " + str(cv))

    scoring = ['accuracy', 'neg_log_loss']
    scores = cross_validate(model, [item[2] for item in mat], [item[1] for item in mat], scoring=scoring, cv=cv, return_train_score=True)

    return scores