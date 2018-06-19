from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from multiscorer import MultiScorer
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def performSVM(training, test):

    model = SVC(C=10000.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=1, gamma='auto', kernel='poly',
    max_iter=-1, probability=True, random_state=42, shrinking=True,
    tol=0.001, verbose=False)

    model.fit([item[2] for item in training], [item[1] for item in training])

    prediction_prob = model.predict_proba([item[2] for item in test])

    prediction = model.predict([item[2] for item in test])

    return prediction, prediction_prob

def performCrossValidationSVM(mat, C=10000):

    model = SVC(C=10000.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=1, gamma='auto', kernel='poly',
    max_iter=-1, probability=True, random_state=42, shrinking=True,
    tol=0.001, verbose=False)

    cv = KFold(n_splits=10)
    print("C value: " + str(C) + " " + str(cv))

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

def make_meshgrid(x, y, h=.002):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = min(x) - 0.05, max(x) + 0.05
    y_min, y_max = min(y) - 0.05, max(y) +  0.05
    print("Scores: " + str(x_min) + str(x_max) + str(y_min) + str(y_max))
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def performSVM2(training, test):

    # Take the first two features
    X = [item[2] for item in training]
    X0 = [item[0] for item in X]
    X1 = [item[1] for item in X]
    X = list(zip(X0, X1))
    y = list()
    for item in training:
        if item[1] == 'FEMALE\n':
            y.append(0)
        else:
            y.append(1)
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C=10.0  # SVM regularization parameter
    models = (SVC(C=10000.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=1, gamma='auto', kernel='poly',
    max_iter=-1, probability=True, random_state=42, shrinking=True,
    tol=0.001, verbose=False),
              svm.LinearSVC(C=C))
    models = (clf.fit(X, y) for clf in models)

    # title for the plots
    titles = ('SVC with linear kernel','LinearSVC (linear kernel)')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    #X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()

def performMultipleSVM(training, test):

    # Take the first two features
    X = [item[2] for item in training]
    X0 = [item[0] for item in X]
    X1 = [item[1] for item in X]
    X = list(zip(X0, X1))
    y = list()
    for item in training:
        if item[1] == 'FEMALE\n':
            y.append(0)
        else:
            y.append(1)
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C=10000.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='poly', degree=3, C=C))
    models = (clf.fit(X, y) for clf in models)

    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    #X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()
