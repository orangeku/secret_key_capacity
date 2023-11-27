import numpy as np
from scipy.optimize import *
import matplotlib.pyplot as plt
def entropy_x(distribution_x):
    # Calculate the entropy of the distribution
    entropy = -np.nansum(distribution_x * np.log2(distribution_x))
    return entropy
H = entropy_x

def entropy_xy(joint_distribution_xy):
    # Calculate the entropy of the joint distribution
    entropy = -np.sum(joint_distribution_xy * np.log2(joint_distribution_xy))
    return entropy

def mutual_information(distribution_x, distribution_y, joint_distribution_xy):
    # Calculate the entropy of the individual distributions
    entropy_x = -np.sum(distribution_x * np.log2(distribution_x))
    entropy_y = -np.sum(distribution_y * np.log2(distribution_y))

    # Calculate the entropy of the joint distribution
    entropy_xy = -np.sum(joint_distribution_xy * np.log2(joint_distribution_xy))

    # Calculate the mutual information
    mutual_information = entropy_x + entropy_y - entropy_xy
    return mutual_information

X_values = np.array([0, 1])
Y_values = np.array([0, 1])

p = 0.8

joint_pmf_XY = np.array([
    [(1-p)/2, p/2],
    [p/2, (1-p)/2]
])


Z_values = np.array([0, 1])

joint_pmf_XYZ = np.zeros((2, 2, 2))
for i, x in enumerate(X_values):
    for j, y in enumerate(Y_values):
        z = x ^ y  # Compute Z as the XOR of X and Y
        joint_pmf_XYZ[i, j, z] = joint_pmf_XY[i, j]

print(joint_pmf_XYZ)

marginal_X = np.sum(joint_pmf_XYZ, axis=(1,2))
marginal_Y = np.sum(joint_pmf_XYZ, axis=(0,2))
marginal_Z = np.sum(joint_pmf_XYZ, axis=(0,1))
marginal_XY = np.sum(joint_pmf_XYZ, axis= 2)
marginal_YZ = np.sum(joint_pmf_XYZ, axis= 0)
marginal_XZ = np.sum(joint_pmf_XYZ, axis= 1)
ex = entropy_xy(marginal_XY)
print(marginal_XY)
print(ex)

def joint(pxy):
    joint_pmf_XY = pxy

    joint_pmf_XYZ = np.zeros((2, 2, 2))
    for i, x in enumerate(X_values):
        for j, y in enumerate(Y_values):
            z = x ^ y  # Compute Z as the XOR of X and Y
            joint_pmf_XYZ[i, j, z] = joint_pmf_XY[i, j]
    return joint_pmf_XYZ

def obj(pxy):
    pxyz = joint(pxy)
    px = np.sum(pxyz, axis=(1, 2))
    py = np.sum(pxyz, axis=(0, 2))
    pz = np.sum(pxyz, axis=(0, 1))
    return H(px) + H(py) - 2 * H(pxy) + H(pz)



def bss(p):
    return np.array([
        [(1-p)/2, p/2],
        [p/2, (1-p)/2]
    ])


fun2 = lambda x: obj(bss(x))
xs = np.linspace(0, 1, 100)
ys = [fun2(x) for x in xs]

pxyz = joint(bss(0.1))
pxy = bss(0.1)
pz = np.sum(pxyz, (0,1))
print(H(pxy))
print(H(pz))

plt.plot(xs, ys)
plt.show()