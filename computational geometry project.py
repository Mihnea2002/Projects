import time
import matplotlib.pyplot as plt
import numpy as np
import math
import random

startScript=time.time()

def dist(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def brute_force_closest_pair(points):
    """Finds the closest pair of points using brute force."""
    n = len(points)
    min_dist = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = dist(points[i], points[j])
            if d < min_dist:
                min_dist = d
                closest_pair = (points[i], points[j])
    return closest_pair, min_dist

def closest_pair_helper(points_sorted_by_x, points_sorted_by_y):
    """Recursively finds the closest pair of points."""
    n = len(points_sorted_by_x)
    if n <= 3:
        return brute_force_closest_pair(points_sorted_by_x)

    mid = n // 2
    mid_point = points_sorted_by_x[mid]

    left_x = points_sorted_by_x[:mid]
    right_x = points_sorted_by_x[mid:]

    left_y = []
    right_y = []
    for point in points_sorted_by_y:
        if point in left_x:
            left_y.append(point)
        else:
            right_y.append(point)

    left_closest_pair, left_min_dist = closest_pair_helper(left_x, left_y)
    right_closest_pair, right_min_dist = closest_pair_helper(right_x, right_y)

    if left_min_dist < right_min_dist:
        closest_pair = left_closest_pair
        min_dist = left_min_dist
    else:
        closest_pair = right_closest_pair
        min_dist = right_min_dist

    strip_points = []
    for point in points_sorted_by_y:
        if abs(point[0] - mid_point[0]) < min_dist:
            strip_points.append(point)

    strip_size = len(strip_points)
    for i in range(strip_size):
        j = i + 1
        while j < strip_size and (strip_points[j][1] - strip_points[i][1]) < min_dist:
            d = dist(strip_points[i], strip_points[j])
            if d < min_dist:
                min_dist = d
                closest_pair = (strip_points[i], strip_points[j])
            j += 1

    return closest_pair, min_dist

def closest_pair(points):
    """Finds the closest pair of points."""
    points_sorted_by_x = sorted(points, key=lambda p: p[0])
    points_sorted_by_y = sorted(points, key=lambda p: p[1])
    return closest_pair_helper(points_sorted_by_x, points_sorted_by_y)

# Generate random points
print("The number of points:")
n = int(input())
print("The lowerbound:")
o=int(input())
print("The upperbound:")
p=int(input())
print("Would you like to see the coordinates of the points?Insert 1 if yes and 0 in no")
m=int(input())
print("Would you like to have integer points(insert 1) or  points with decimals(insert 0)? *Note that for integers at big number of tpoints the points might reapeat")
t=int(input())
if(t==1):
    points = [(random.randint(o, p), random.randint(o, p)) for i in range(n)]
else:
    points = [(random.uniform(o, p), random.uniform(o, p)) for i in range(n)]
#for float values
# Find the closest pair of points
closest_pair, min_dist = closest_pair(points)
if(m==1):
    print("The points are:",points)
print("Closest pair:", closest_pair)
print("Distance:", min_dist)

# Visualize the points and the closest pair
x, y = zip(*points)
plt.scatter(x, y)
plt.plot([closest_pair[0][0], closest_pair[1][0]], [closest_pair[0][1], closest_pair[1][1]], 'r')
plt.show()
endScript=time.time()
timeofScript=startScript-endScript
print(str(timeofScript)+"seconds")

