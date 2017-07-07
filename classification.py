import pandas as pd

up = pd.read_csv('./datasets/gabo/up_training.csv')

X = up.iloc[:,2:-2]
y = up.iloc[:,-1]

X['lear'] = X['lear'].astype('category')
X['rear'] = X['rear'].astype('category')

y = y.astype('category')

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
