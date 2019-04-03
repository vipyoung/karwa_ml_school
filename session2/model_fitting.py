import requests
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ParameterGrid
from collections import defaultdict
from random import shuffle
import sys 


def mape_score(y_true, y_pred):
	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def find_best_parameters(X, Y):
	gbt_params = {'n_estimators': [10, 20, 50, 100, 200, 250],
	              'max_features': ["auto", "sqrt", "log2"],
	              'max_depth': [3, 5, 10, 15, 20, 25],
	              'warm_start': [False, True],
	              }
	gbr = GradientBoostingRegressor(random_state=0, learning_rate=0.1)
	best_mape = 100000
	for g in ParameterGrid(gbt_params):
		gbr.set_params(**g)
		gbr.fit(X[:len(X)*8/10], Y[:len(Y)*8/10])
		gbr_y = gbr.predict(X[len(X)*8/10:])
		actual_mape = mape_score(Y[len(Y)*8/10:], gbr_y)
		# save if best
		if  actual_mape < best_mape:
			best_mape = actual_mape
			best_grid = g
		print actual_mape, g
	print 'Best MAPE:', best_mape
	print 'Best params:', best_grid
	return best_grid

mapes = []
maes = []
nb_trips = []
mean_krws = []
mean_osrms=[]
#
# for hour in range(24):
coordinates = []
hours = []
data = []

assert len(sys.argv) == 3, 'Make sure you run: python model_fitting.py city features. city in [doha, porto, nyc], features in [ta, sta]'
city = sys.argv[1]
assert city in ['doha', 'porto', 'nyc'], 'City should be in [doha, porto, nyc]'
features = sys.argv[2]
assert features in ['ta', 'sta'], 'feature should be in [ta, sta]'

with open('data/%s_zoned.csv' % city) as f:
	f.readline()
	k_dist, k_dur, o_dist, o_dur = [], [], [], []
	for line in f:
		x = line.strip().split(',')
		# discard cases where trips run in zero seconds/minutes
		if float(x[6]) == 0:
			continue
		slon, slat, dlon, dlat = map(float, x[1:5])
                # Filter wrong data for Doha only:
		# Qatar bbox: 50.7211, 24.4708, 51.7126, 26.2156
                if (city == 'doha') and (slon > 51.71 or slon < 50.7211 or dlon > 51.71 or dlon < 50.7211 or \
			slat > 26.2156 or slat < 24.4708 or dlat > 26.2156 or dlat < 24.4708):
			continue

		dt = datetime.strptime(x[0], '%m-%d-%y %H:%M')
		hour_of_day = dt.hour
		day_of_week = dt.weekday()

		# For the off-traffic stats
		# if hour_of_day >5:
		# # For the morning commute stats
		# if hour_of_day < 6 or hour_of_day > 9:
		# 	continue
		# # For the hourly breakdown
		# # if hour_of_day != hour:
		# 	continue

		hours.append(hour_of_day+1)
		k_dur = float(x[6])
		o_dist = float(x[7])
		o_dur = float(x[8])
		s_zone = x[9]
		d_zone = x[10]
                if features == 'sta':
                    # STA-Model: spatio-temporal
		    # data.append([hour_of_day+1, day_of_week+1, day_of_week*24+hour_of_day, o_dist, int(s_zone), int(d_zone), o_dur, k_dur])
		    data.append([hour_of_day+1, day_of_week+1, day_of_week*24+hour_of_day, o_dist, s_zone, d_zone, o_dur, k_dur])
                elif features == 'ta':
                    # TA-Model: temporal only
                    data.append([hour_of_day+1, day_of_week+1, day_of_week*24+hour_of_day, o_dist, o_dur, k_dur])
		coordinates.append([dlon, dlat])
                

X0 = [_[:-1] for _ in data]
Y0 = [_[-1] for _ in data]

# Shuffle the data
shuf = range(len(data))
shuffle(shuf)
X, Y = [], []
for i in shuf:
	X.append(X0[i])
	Y.append(Y0[i])

# best_model_params = find_best_parameters(X, Y)
## Best model: {'max_features': 'sqrt', 'n_estimators': 100, 'warm_start': False, 'max_depth': 5}

lim = int(len(data)*.7)
mod = GradientBoostingRegressor(n_estimators=100, max_depth=5)
mod.fit(X[:lim], Y[:lim])
yhat = mod.predict(X[lim:])

y_osrm = np.array([_[-1] for _ in X[lim:]])

# Per hour comparison of mapes:
hours = [h[0]-1 for h in X[lim:]]
gt = Y[lim:]

gt_hourly_pred = defaultdict(list)
osrm_hourly_pred = defaultdict(list)
optimized_osrm_hourly_pred = defaultdict(list)

for i, h in enumerate(hours):
	gt_hourly_pred[h].append(gt[i])
	osrm_hourly_pred[h].append(y_osrm[i])
	optimized_osrm_hourly_pred[h].append(yhat[i])

print "HOURLY MAE"
for h in sorted(gt_hourly_pred.keys()):
	#print h, ',', mape_score(np.array(gt_hourly_pred[h]), np.array(osrm_hourly_pred[h])), ',', mape_score(np.array(gt_hourly_pred[h]), np.array(optimized_osrm_hourly_pred[h]))
	print h, ',', mean_absolute_error(np.array(gt_hourly_pred[h]), np.array(osrm_hourly_pred[h])), ',', mean_absolute_error(np.array(gt_hourly_pred[h]), np.array(optimized_osrm_hourly_pred[h]))

print 'HOURLY MAPE'
for h in sorted(gt_hourly_pred.keys()):
	print h, ',', mape_score(np.array(gt_hourly_pred[h]), np.array(osrm_hourly_pred[h])), ',', mape_score(np.array(gt_hourly_pred[h]), np.array(optimized_osrm_hourly_pred[h]))

print 'OSRM:',
print "MAPE: %s, MAE: %s" % (mape_score(Y[lim:], y_osrm), mean_absolute_error(Y[lim:], y_osrm))
print 'Fitted OSRM:',
print "MAPE: %s, MAE: %s" % (mape_score(Y[lim:], yhat), mean_absolute_error(Y[lim:], yhat))



##################
# Plot maps
##################
#
# def get_color(d):
# 	if d < 3:
# 		return 'green'
# 	if d < 5:
# 		return 'yellow'
# 	if d < 10:
# 		return 'orange'
# 	return 'red'
#
# colors = []
# X = []
# Y = []
# for i, d in enumerate(dur_diff):
# 	if np.abs(d) < 5 or np.abs(d) >=10 :
# 		continue
# 	colors.append(get_color(np.abs(d)))
# 	X.append(coordinates[i][0])
# 	Y.append(coordinates[i][1])
#
# plt.figure(figsize=[12,12])
# #plt.hexbin(X, Y, gridsize=200)
# plt.scatter(X, Y, color=colors)
# plt.xlim(xmin=51.30, xmax=max(X))
# plt.ylim([25.10, 25.45])
# #plt.axis('equal')
# #plt.legend(['0-3 mins', '3-5 mins', '5-10 mins', '10+ mins' ])
# plt.savefig('figs/geo_karwa-osrm_duration_cummute_5-10.png', FORMAT='PNG')
# plt.show()


##################
# Plot histograms
##################
# plt.figure(figsize=[12, 9])
# plt.hist(dur_diff, bins='auto')
# plt.xlim(xmax=80, xmin=-80)
# plt.xlabel('KARWA_ETA - OSRM_ETA')
# plt.ylabel("# Trips")
# plt.savefig('figs/karwa-osrm-duration-06_to_09am.png', FORMAT='PNG')
# plt.show()


######################
# Plot Scatter Plot
#####################
# plt.figure(figsize=[12, 9])
# plt.scatter(k_dist, o_dist)
# plt.plot(range(int(min(max(k_dist),max(o_dist)))), lw=2, color='red')
# plt.ylim(ymin=0)
# plt.xlim(xmin=0)
# plt.xlabel('Karwa Duration (mins)')
# plt.ylabel('OSRM Duration (mins)')
# plt.savefig('figs/karwa-osrm-duration-06_to_09am_scatter.png', FORMAT='PNG')
# plt.show()
