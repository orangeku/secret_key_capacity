#! /usr/bin/python

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import hull

fig = plt.figure() # For plotting
ax = fig.add_subplot(111)

for point in hull.final_vertices:
	ax.scatter(point.x, point.y, c='b', marker='o')

ax.set_xlabel('alpha Label')
ax.set_ylabel('beta Label')


plt.savefig('image.jpg', bbox_inches='tight')
plt.show()
