from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def performSVM(training, test):

    model = SVC(C=10000.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=1, gamma='auto', kernel='poly',
    max_iter=-1, probability=True, random_state=42, shrinking=True,
    tol=0.001, verbose=False)

    model.fit([item[2] for item in training], [item[1] for item in training])

    prediction_prob = model.predict_proba([item[2] for item in test])

    prediction = model.predict([item[2] for item in test])

    return prediction, prediction_prob

def performCrossValidationSVM(mat):

    model = SVC(C=10000.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=1, gamma='auto', kernel='poly',
    max_iter=-1, probability=True, random_state=42, shrinking=True,
    tol=0.001, verbose=False)

    #pipeline = Pipeline([('transformer', scalar), ('estimator', clf)])

    cv = KFold(n_splits=40)
    #print("K value: " + str(cv))
    scores = cross_val_score(model, [item[2] for item in mat], [item[1] for item in mat], cv=cv,  scoring='neg_log_loss')

    return scores