import pandas as pd

orientation = "left"
up = pd.read_csv('./datasets/gabo/{}_training.csv'.format(orientation))

X = up.iloc[:,2:-2]
y = up.iloc[:,-1]


#One hot enconding categorical features
X['lear'] = X['lear'].astype('category')
X['rear'] = X['rear'].astype('category')
X['lear'] = pd.get_dummies(X["lear"], drop_first=True)
X['rear'] = pd.get_dummies(X["rear"], drop_first=True)
# y = y.astype('category')
y = y == orientation
y = pd.get_dummies(y, drop_first=True)

#Scaling numerical values
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X[X.columns.drop(['lear', 'rear'])] = sc_X.fit_transform(X[X.columns.drop(['lear', 'rear'])])

# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

#Split into train,test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

#Fitting the model
from sklearn.tree import DecisionTreeClassifier, export_graphviz
classifier = DecisionTreeClassifier(presort=True)
model = classifier.fit(X_train,y_train)


#Saving tree image
import pydotplus
dot_data = export_graphviz(model, out_file=None,
                           feature_names=list(X),
                           class_names=['Not '+orientation, orientation],
                           filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("tree_{}.pdf".format(orientation))

#SVM model
from sklearn.svm import SVC
svc_classifier = SVC()
svc_classifier.fit(X_train,y_train)


#Saving the model to use later
from sklearn.externals import joblib
joblib.dump(model, "up_tree_model.pkl")
joblib.dump(sc_X, "up_tree_scaler.pkl")
