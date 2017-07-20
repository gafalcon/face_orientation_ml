import pandas as pd
import numpy as np
from datetime import datetime

from os import listdir, mkdir
from os.path import isfile, join
import sys

if len(sys.argv) < 2:
    print "Ingrese ubicacion de archivos csv"
    sys.exit(-1)

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

#Split into train,test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


#Fitting the model
from sklearn.tree import DecisionTreeClassifier, export_graphviz
classifier = DecisionTreeClassifier(presort=True)#, max_depth=10)
model = classifier.fit(X_train,y_train)

#Evaluating the model
from sklearn.metrics import classification_report
print ("Model score: ", model.score(X_test, y_test))

y_predicted = model.predict(X_test)
print(classification_report(y_test, y_predicted,target_names=classes))
# for i, class_name in enumerate(classes):
#     features_from_ith_class = X_test[y_test == i]
#     print ( "Score for class", class_name, ":",  model.score(features_from_ith_class, np.ones((len(features_from_ith_class),), dtype=np.int8)*i))

#Saving tree image
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
