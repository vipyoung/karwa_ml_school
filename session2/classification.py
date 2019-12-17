from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import csv


fname ='matCCC2.csv'

X, y = [], []
with open(fname) as f:
    f.readline()
    for line in f:
        #o = line.split(';')
        o = line.strip().replace(',', '.').split(';')
        X.append(list(map(float, o[:-1])))
        y.append(int(o[-1]))
algorithm = [LogisticRegression(), DecisionTreeClassifier(), GaussianNB(), RandomForestClassifier(n_estimators=10,
                                                                                                  max_depth=5), GradientBoostingClassifier(),SVC()]
algorithm_name = ['LogisticRegression', 'DecisionTreeClassifier', 'GaussianNB', 'RandomForestClassifier','GradientBoostingClassifier', 'SVC']

for idx, mod in enumerate(algorithm):
    score = []
    precision = []
    recall = []
    f1 = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        # fit the model with data
        mod.fit(X_train, y_train)
        #print('Coefficients:', mod.coef_)
        y_pred = mod.predict(X_test)
        score.append(mod.score(X_test, y_test))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred))
    srore_moyen = round(float(sum(score))/len(score), 2)
    precision_moyen = round(float(sum(precision))/len(precision), 2)
    recall_moyen = round(float(sum(recall))/len(recall), 2)
    f1_moyen = round(float(sum(f1)) / len(f1), 2)
    print('ALGORITHME:', algorithm_name[idx])
    print('\tsrore_moyen', srore_moyen)
    print('\tprecision_moyen', precision_moyen)
    print('\trecalle_moyen', recall_moyen)
    print('\tf1_score', f1_moyen)
'''''
with open('result.csv', 'a', newline='') as f1:
    spamwriter = csv.writer(f1, delimiter=';')
    spamwriter.writerow(['LogisticRegression', str(srore_moyen), str(precision_moyen), str(recall_moyen)])
'''''
'''''
# pour la prédiction d'une seule valeur ou une ligue
 y_pred = mod.predict([[1,0,2,5,.....]])
# x_test doit être une liste de listes cad [[ ]] 
'''''