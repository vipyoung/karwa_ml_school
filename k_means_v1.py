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
    kmeans = KMeans(n_clusters=4, random_state=1).fit(data)
    print kmeans.inertia_
    labels = kmeans.labels_
   
    # Plot clusters
    X = [p[0] for p in points]
    Y = [p[1] for p in points]
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'cyan']
    cs = [colors[labels[i]] for i in range(len(points))]
    plt.scatter(X,Y, c=cs)
    #Plot centers
    centers = kmeans.cluster_centers_
    plt.scatter([p[0] for p in centers], [p[1] for p in centers], marker='*', s=100)
   
    plt.axis('equal')
    plt.show()
