from matplotlib import pyplot as plt
from datetime import datetime
import math
import random
from geopy.distance import great_circle

def euclidean(a, b):
    return math.sqrt(sum([(a[0]-b[0])**2, (a[1] - b[1])**2]))

def distance(a, b):
    """
    Points should be supplied as (lon,lat), they will then be switched to lat,lon
    """
    return great_circle((a[1], a[0]), (b[1], b[0])).meters


def assignment(points, seeds):
    clusters = [[] for _ in range(len(seeds))]
    for i, point in enumerate(points):
        min_dist = float('inf')
        assignment_id = -1
        for j, seed in enumerate(seeds):
            dist = euclidean(seed, point)
            if dist < min_dist:
                min_dist = dist
                assignment_id = j
        assert assignment_id != -1, 'Problem with assignment'
        clusters[assignment_id].append(i)
    return clusters

def update_centers(clusters, points):
    centers = []
    for cls in clusters:
        x_c = sum([points[p][0] for p in cls])/len(cls)
        y_c = sum([points[p][1] for p in cls])/len(cls)
        centers.append((x_c,y_c))
    return centers

def k_means(k, points, max_iter=10):
    seed_indexes = random.sample(range(len(points)), k)
    seeds = [points[idx] for idx in seed_indexes]
    centers =  seeds
    nb_iter = 0
    while True:
        nb_iter += 1
        print nb_iter
        old_centers = centers
        clusters = assignment(points, seeds)
        centers = update_centers(clusters, points)
        if (old_centers == centers) or (nb_iter == max_iter):
            break
    return clusters


if __name__ == '__main__':
    # Read points
    points = []
    with open('data/pure_gps.csv') as f:
        for line in f:
            lat,lon = line.split(',')
            lat = float(lat)
            lon = float(lon)
            points.append((lon,lat))

    # Run clustering
    clusters = k_means(k=4, points=points, max_iter=10)

    # Plot assignments. 
    point_to_cluster = dict()
    for i, cls in enumerate(clusters):
        print 'Len cls:', i, len(cls)
        for j in cls:
            point_to_cluster[j] = i    
    X = [p[0] for p in points]
    Y = [p[1] for p in points]
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'cyan']
    cs = [colors[point_to_cluster[i]] for i in range(len(points))]
    plt.scatter(X,Y, c=cs)
    # Plot centers
    centers = update_centers(clusters, points)
    plt.scatter([p[0] for p in centers], [p[1] for p in centers], marker='*', s=100)
   
    plt.axis('equal')
    plt.show()
