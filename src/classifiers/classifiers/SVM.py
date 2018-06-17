from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def performSVM(training, test, mat):

    #scalar = StandardScaler()
    model = SVC(C=10000.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=1, gamma='auto', kernel='poly',
    max_iter=-1, probability=True, random_state=42, shrinking=True,
    tol=0.001, verbose=False)

    #pipeline = Pipeline([('transformer', scalar), ('estimator', clf)])

    #tfidf = TfidfVectorizer()

    #vect_data = tfidf.fit_transform(np.array([item[2] for item in training]))
    #vect_data1 = tfidf.fit_transform([item[2] for item in training])


    #cv = KFold(n_splits=40)
    #print("K value: " + str(cv))
    #scores = cross_val_score(model, [item[2] for item in mat], [item[1] for item in mat], cv=cv,  scoring='neg_log_loss')
    #print("Scores "+ str(scores))
    #print("Neg_log_loss: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))

    #model = LinearSVC(C=100.0, random_state=42)
    #model = LinearSVC(loss='l2', penalty='l1', dual=False)
    #model = CalibratedClassifierCV(model)
    model.fit([item[2] for item in training], [item[1] for item in training])
    prediction_prob = model.predict_proba([item[2] for item in test])
    #print("prediction " + str(prediction_prob))
    prediction = model.predict([item[2] for item in test])
    #scores = cross_val_score(prediction, mat, [item[2] for item in training], cv=5)
    #scores

    return prediction, prediction_prob