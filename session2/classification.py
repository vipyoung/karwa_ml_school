from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
fname = 'diabetes.csv'
X, y = [], []
with open(fname) as f:
    f.readline()
    for line in f:
        o = line.split(',')
        X.append(map(float, o[:-1]))
        y.append(int(o[-1]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# instantiate model
mod = LogisticRegression()

# fit the model with data
mod.fit(X_train,y_train)
print 'Coefficients:', mod.coef_
#
y_pred = mod.predict(X_test)
print 'Score:', mod.score(X_test, y_test)
print 'Precision:', precision_score(y_test, y_pred)
print 'Recall:', recall_score(y_test, y_pred)
