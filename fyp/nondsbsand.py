import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

X_values = np.array([0, 1])
Y_values = np.array([0, 1])
Z_values = np.array([0, 1])

def entropy_x(distribution_x):
    # Calculate the entropy of the distribution
    entropy = -np.nansum(distribution_x * np.log2(distribution_x))
    return entropy
H = entropy_x
pxy = np.array([
        [0.5, 0.01],
        [0.01, 0.48]
    ])
px = np.array([0.51, 0.49])
py = np.array([0.51, 0.49])
pz = np.array([0.52, 0.48])
print(H(px))
print(H(py))
print(H(pz))
print(H(pxy))
print(H(px) + H(py) - 2 * H(pxy) + H(pz))

def bss(p, q):
    return np.array([
        [p, q],
        [q, 1-p-2*q]
    ])

def joint(pxy):
    joint_pmf_XY = pxy

    joint_pmf_XYZ = np.zeros((2, 2, 2))
    for i, x in enumerate(X_values):
        for j, y in enumerate(Y_values):
            z = x ^ y  # Compute Z as the AND of X and Y
            joint_pmf_XYZ[i, j, z] = joint_pmf_XY[i, j]
    return joint_pmf_XYZ

def obj(p,q):
    if 1 - p - 2*q <=0:
        return None     
    pxy = np.array([
        [p, q],
        [q, 1-p-2*q]
    ])
    pxyz = joint(pxy)
    px = np.sum(pxyz, axis=(1, 2))
    py = np.sum(pxyz, axis=(0, 2))
    pz = np.sum(pxyz, axis=(0, 1))
    return H(px) + H(py) - 2 * H(pxy) + H(pz)
def obj1(p,q):
    if 1 - p - 2*q <=0:
        return None     
    pxy = np.array([
        [p, q],
        [q, 1-p-2*q]
    ])
    pxyz = joint(pxy)
    px = np.sum(pxyz, axis=(1, 2))
    print(px)
    py = np.sum(pxyz, axis=(0, 2))
    print(py)
    pz = np.sum(pxyz, axis=(0, 1))
    print(pz)
    return H(px) + H(py) - 2 * H(pxy) + H(pz)
a=obj1(0.1, 0.2)
print(a)

fun2 = lambda x,y: obj(x, y)
xs = np.linspace(0, 1, 100)
ys = np.linspace(0, 1, 100)
zs = [fun2(x,y) for x in xs for y in ys]
zs = np.array(zs)
zs = zs.reshape((100,100))
np.set_printoptions(threshold=10001)
xs = np.outer(np.linspace(0, 1, 100), np.ones(100))
ys = np.outer(np.linspace(0, 1, 100), np.ones(100))
ys = ys.T

#print(xs)
#print(ys)
#print(zs)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xs, ys, zs, cmap='viridis')
ax.set_xlabel('Alpha')
ax.set_ylabel('Beta')
ax.set_zlabel('Value')
ax.set_title('3D surface')
plt.ylim(0, 0.5)
plt.show()






"""
# Define the function
def f(x, y):
    
    if (1 - x - 2*y) <= 0 or x == 0 or y == 0:
        return 0
    else:
        #calculate H(X) + H(Y) - 2H(XY) + H(Z)
        return 2*(-(x+y)*math.log2(x+y)-(1-x-y)*math.log2(1-x-y)) - 2*(-x*math.log2(x)-2*y*math.log2(y)-(1-x-2*y)*math.log2(1-x-2*y)) + (-(x+2*y)*math.log2(x+2*y)-(1-x-2*y)*math.log2(1-x-2*y))
f = np.vectorize(f)
# Define the range of X and Y
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

X, Y = np.meshgrid(x, y)


Z = f(X, Y)
np.set_printoptions(threshold=10001)
print(Z)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('Alpha')
ax.set_ylabel('Beta')
ax.set_zlabel('Value')
ax.set_title('3D surface')
plt.show()
"""