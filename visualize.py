from matplotlib import pyplot as plt


points = []
with open('data/pure_gps.csv') as f:
    for line in f:
        lat,lon = line.split(',')
        lat = float(lat)
        lon = float(lon)
        points.append((lon,lat))
X = [p[0] for p in points]
Y = [p[1] for p in points]
plt.scatter(X,Y)
plt.axis('equal')
plt.show()


"""
with open('data/gps.csv') as f:
    for line in f:
        vid,dt,lat,lon,speed,ang = line.split(',')
        lat = float(lat)
        lon = float(lon)
        speed = float(speed)
        ang = float(ang)
        points.append((lon,lat))
"""
