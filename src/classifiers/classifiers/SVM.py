from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt


def performSVM(training, test):

    model = SVC(C=10000.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=1, gamma='auto', kernel='linear',
    max_iter=-1, probability=True, random_state=42, shrinking=True,
    tol=0.001, verbose=False)

    model.fit([item[2] for item in training], [item[1] for item in training])

    prediction_prob = model.predict_proba([item[2] for item in test])

    prediction = model.predict([item[2] for item in test])

    return prediction, prediction_prob

def performCrossValidationSVM(mat, C=10000):

    model = SVC(C=C, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=1, gamma='auto', kernel='linear',
    max_iter=-1, probability=True, random_state=42, shrinking=True,
    tol=0.001, verbose=False)

    cv = KFold(n_splits=10)
    print("C value: " + str(C) + " " + str(cv))

    scoring = ['accuracy', 'neg_log_loss']
    scores = cross_validate(model, [item[2] for item in mat], [item[1] for item in mat], scoring=scoring, cv=cv, return_train_score=True)

    return scores


def make_meshgrid(x, y, h=.002):

    x_min, x_max = min(x) - 0.05, max(x) + 0.05
    y_min, y_max = min(y) - 0.05, max(y) +  0.05
    print("Scores: " + str(x_min) + str(x_max) + str(y_min) + str(y_max))
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def performMultipleSVM(training, test):

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

    models = (SVC(C=10, kernel='linear'),
              SVC(C=200, kernel='linear'))
    models = (clf.fit(X, y) for clf in models)

    titles = ('Linear SVC C=10','Linear SVC C=200')

    fig, sub = plt.subplots(2, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('DIM 1')
        ax.set_ylabel('DIM 2')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()
