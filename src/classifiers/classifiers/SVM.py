from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from multiscorer import MultiScorer
from sklearn.metrics import confusion_matrix


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

    cv = KFold(n_splits=10)
    print("K value: " + str(cv))

    scoring = ['accuracy', 'neg_log_loss']
    scores = cross_validate(model, [item[2] for item in mat], [item[1] for item in mat], scoring=scoring, cv=cv, return_train_score=True)

    return scores


def performCrossValidationSVM2(mat):

    model = SVC(C=10000.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=1, gamma='auto', kernel='poly',
    max_iter=-1, probability=True, random_state=42, shrinking=True,
    tol=0.001, verbose=False)

    #pipeline = Pipeline([('transformer', scalar), ('estimator', clf)])

    cv = KFold(n_splits=10)
    print("K value: " + str(cv))
    scorer = MultiScorer({
        'accuracy': (accuracy_score, {}),
    #    'precision': (precision_score, {}),
    #    'recall': (recall_score, {}),
        'neg_log_loss': (log_loss, {})
    })
    scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro'}

    #for train_index, test_index in cv.split(X):
    #    model.fit(X[train_index], labels[train_index])
    #    ypred = model.predict(X[test_index])
    #    kappa_score = cohen_kappa_score(labels[test_index], ypred)
    #    confusion_matrix = confusion_matrix(labels[test_index], ypred)

    scores = cross_val_score(model, [item[2] for item in mat], [item[1] for item in mat], cv=cv, scoring=scoring)
    #results = scorer.get_results()
    #print("Results: "+str(results))
    print("Scores: " + str(scores.keys))
    return scores