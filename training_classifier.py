import pandas as pd
import numpy as np
from datetime import datetime

from os import listdir, mkdir
from os.path import isfile, join
import sys

#importing classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz

save_models = True

if len(sys.argv) < 2:
    print "Ingrese ubicacion de archivos csv"
    sys.exit(-1)
if len(sys.argv) == 3 and sys.argv[2]:
    save_models = False

filepath = sys.argv[1]

print "Looking for csv files in ", filepath
csvfiles = [join(filepath,f) for f in listdir(filepath) if f.endswith(".csv")]
print "files found", csvfiles

df = pd.concat([pd.read_csv(csvfile) for csvfile in csvfiles])

print "Features", df.columns
print "Num of training rows", len(df)

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#One hot enconding categorical features
X['lear'] = X['lear'].astype('category')
X['rear'] = X['rear'].astype('category')
X['lear'] = pd.get_dummies(X["lear"], drop_first=True)
X['rear'] = pd.get_dummies(X["rear"], drop_first=True)

y = y.astype("category")
classes = list(y.cat.categories)
y = y.cat.codes

#Scaling numerical values
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X[X.columns.drop(['lear', 'rear'])] = sc_X.fit_transform(X[X.columns.drop(['lear', 'rear'])])

# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

#Cross validation
from sklearn.model_selection import cross_val_score, ShuffleSplit

for C in [60., 70., 75., 90.]:
    svm_classifier = SVC(C=C, kernel='rbf')
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(svm_classifier, X, y, cv=cv)
    print "Scores with C=",C, scores
    print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

#Best C according to cross-validati0n
C = 70.

for max_depth in [None, 10, 20, 15]:
    tree_classifier = DecisionTreeClassifier(max_depth=max_depth)
    cv = ShuffleSplit(n_splits=5, test_size=0.3)
    scores = cross_val_score(tree_classifier, X, y, cv=cv)
    print "Scores with max_depth=",max_depth, scores
    print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

#Best max_depth
max_depth = 15

for n_estimators in [10, 15, 20]:
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    cv = ShuffleSplit(n_splits=5, test_size=0.3)
    scores = cross_val_score(rf_classifier, X, y, cv=cv)
    print "Scores with n_estimators=",n_estimators, scores
    print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

#Best n_estimators
n_estimators = 20

#Split into train,test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


#Fitting the model
classifier = DecisionTreeClassifier(presort=True, max_depth=max_depth)
model = classifier.fit(X_train,y_train)

svm_classifier = SVC(C=C, kernel='rbf', probability=True)
svm_model = svm_classifier.fit(X_train,y_train)

forest_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
forest_model = forest_classifier.fit(X_train, y_train)

#Evaluating the model
from sklearn.metrics import classification_report
print ("Report for tree classifier")
print ("Tree Model score: ", model.score(X_test, y_test))
y_predicted = model.predict(X_test)
print(classification_report(y_test, y_predicted,target_names=classes))

print ("Report for svm classifier")
print ("svm Model score: ", svm_model.score(X_test, y_test))
y_predicted = svm_model.predict(X_test)
print(classification_report(y_test, y_predicted,target_names=classes))
# for i, class_name in enumerate(classes):
#     features_from_ith_class = X_test[y_test == i]
#     print ( "Score for class", class_name, ":",  model.score(features_from_ith_class, np.ones((len(features_from_ith_class),), dtype=np.int8)*i))


print ("Report for Random Forest classifier")
print ("Random Forest Model score: ", forest_model.score(X_test, y_test))
y_predicted = forest_model.predict(X_test)
print(classification_report(y_test, y_predicted,target_names=classes))

#Saving tree image
if save_models:
    timestamp = datetime.now().strftime("%d.%m_%H.%M.%S")
    import pydotplus
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=list(X),
                               class_names=classes,
                               filled=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(join("results","tree_model_full_"+timestamp+".pdf"))

    #Saving the model to use later
    from sklearn.externals import joblib
    joblib.dump(model, join("results","tree_model_"+timestamp+".pkl"))
    joblib.dump(sc_X, join("results","tree_scaler_"+timestamp+".pkl"))
    joblib.dump(forest_model, join("results", "random_forest_model_"+timestamp+".pkl"))
    joblib.dump(svm_model, join("results", "svm_model_"+timestamp+".pkl"))


#Precision-Recall curves
# from sklearn.metrics import precision_recall_curve
# y_score = svm_model.decision_function(X_test)
# precision, recall, thresholds = precision_recall_curve(y_test, y_score)
# from matplotlib import pyplot as plt
# plt.plot(recall, precision)
# plt.show()
