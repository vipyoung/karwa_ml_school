from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import sys

if __name__ == '__main__':
    # Read data points
    points = []
    with open('data/pure_gps.csv') as f:
        for line in f:
            lat,lon = line.split(',')
            lat = float(lat)
            lon = float(lon)
            points.append((lon,lat))
    data = np.array(points)
    
    # Run clustering
    scores = []
    for k in range(1,8):
        kmeans = KMeans(n_clusters=k, random_state=1).fit(data)
        scores.append(kmeans.inertia_)
    plt.plot(range(1,8), scores)
    plt.show()
