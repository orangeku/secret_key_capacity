
# importing libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
 
# defining surface and axes
x = np.outer(np.linspace(-2, 2, 3), np.ones(3))
y = x.copy().T
z = np.cos(x ** 2 + y ** 3)
print(x)
print(y)
print(z)
 
fig = plt.figure()
 
# syntax for 3-D plotting
ax = plt.axes(projection='3d')
 
# syntax for plotting
ax.plot_surface(x, y, z, cmap='viridis',\
                edgecolor='green')
ax.set_title('Surface plot geeks for geeks')
plt.show()
