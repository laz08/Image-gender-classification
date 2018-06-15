from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


def performSVM(training, test, mat):

    #scalar = StandardScaler()
    #clf = LinearSVC(C=100.0, random_state=42)

    #pipeline = Pipeline([('transformer', scalar), ('estimator', clf)])

    #tfidf = TfidfVectorizer()

    #vect_data = tfidf.fit_transform(np.array([item[2] for item in training]))
    #vect_data1 = tfidf.fit_transform([item[2] for item in training])


    #cv = KFold(n_splits=40)
    #print("K value: " + str(cv))
    #scores = cross_val_score(clf, [item[2] for item in mat], [item[1] for item in mat], cv = cv)
    #print("Scores "+ str(scores))
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))

    model = LinearSVC(C=100.0, random_state=42)
    #model = LinearSVC(loss='l2', penalty='l1', dual=False)
    model.fit([item[2] for item in training], [item[1] for item in training])

    prediction = model.predict([item[2] for item in test])
    #scores = cross_val_score(prediction, mat, [item[2] for item in training], cv=5)
    #scores

    return prediction
    
    