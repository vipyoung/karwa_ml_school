from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ParameterGrid
from collections import defaultdict
from random import shuffle
import sys 


def mape_score(y_true, y_pred):
	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def read_data_distance(fname):
    data = []
    with open(fname) as f:
        f.readline()
        for line in f:
            x = line.strip().split(',')
            # discard cases where trips run in zero seconds/minutes
            if float(x[6]) == 0:
                continue
            # Filter wrong data for Doha only. Qatar bbox: 50.7211, 24.4708, 51.7126, 26.2156
            slon, slat, dlon, dlat = map(float, x[1:5])
            if slon > 51.71 or slon < 50.7211 or dlon > 51.71 or dlon < 50.7211 or \
                slat > 26.2156 or slat < 24.4708 or dlat > 26.2156 or dlat < 24.4708:
                continue
            krw_dist = float(x[5])
            krw_dur = float(x[6])
            osrm_dist = float(x[7])
            osrm_dur = float(x[8])
            data.append([osrm_dist, osrm_dur, krw_dur])
    return data



if __name__ == '__main__':
    # 1. Read data
    # fname = 'doha_zoned.csv'
    fname = 'karwa_osrm_eta.csv'
    data = read_data_distance(fname)
    print '# observations:', len(data)
    
    # 2. Shuffle data
    shuffle(data)
    X = [_[:-1] for _ in data]
    Y = [_[-1] for _ in data]
    print 'Data shuffled'
    # 3. Split data: Training/Test
    lim = int(len(data)*.7)
    X_train, X_test = X[:lim], X[lim:]
    Y_train, Y_test = Y[:lim], Y[lim:]

    # 4. Prepare and fit model
    mod = LinearRegression()
    # mod = SVR(gamma='scale', C=1.0, epsilon=0.2)
    # mod = RandomForestRegressor(n_estimators=50, max_depth=4)
    # mod = GradientBoostingRegressor(n_estimators=100, max_depth=5)
    mod.fit(X_train, Y_train)
    print 'Model fitted'
    print 'coefficients:', mod.coef_

    # 5. Predict
    Y_predict = mod.predict(X_test)
    print 'Prediction computed'

    # y_osrm = np.array([_[-1] for _ in X[lim:]])

    # Per hour comparison of mapes:
    # hours = [h[0]-1 for h in X[lim:]]

    print 'Mean Absolute Error:', mean_absolute_error(Y_test, Y_predict) 
    print 'Mean Absolute Percentage Error:', mape_score(Y_test, Y_predict)
    plt.scatter(Y_test, Y_predict)
    plt.xlim([0, 120])
    plt.ylim([0, 120])
    plt.show()
