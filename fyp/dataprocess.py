import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

f = open("dataxor.txt", "a")
X_values = np.array([0, 1])
Y_values = np.array([0, 1])
Z_values = np.array([0, 1])
def joint(pxy):
    joint_pmf_XY = pxy

    joint_pmf_XYZ = np.zeros((2, 2, 2))
    for i, x in enumerate(X_values):
        for j, y in enumerate(Y_values):
            z = x ^ y  # Compute Z as the AND of X and Y
            joint_pmf_XYZ[i, j, z] = joint_pmf_XY[i, j]
    return joint_pmf_XYZ

def entropy_x(distribution_x):
    # Calculate the entropy of the distribution
    entropy = -np.nansum(distribution_x * np.log2(distribution_x))
    return entropy
H = entropy_x
def obj1(p,q):
    if 1 - p - 2*q <=0:
        return None     
    pxy = np.array([
        [p, q],
        [q, 1-p-2*q]
    ])
    pxyz = joint(pxy)
    px = np.sum(pxyz, axis=(1, 2))
    #print(px)
    py = np.sum(pxyz, axis=(0, 2))
    #print(py)
    pz = np.sum(pxyz, axis=(0, 1))
    #print(pz)
    return H(px) + H(py) - 2 * H(pxy) + H(pz)

count = 0


for p in range(1, 99, 1):
    for q in range(1, 99, 1):
        if 1 - (p/100) - 2 * (q/100) > 0:
            result = obj1(p/100,q/100)
            f.writelines(str(p/100) + " " + str(q/100) + " " + str(result))
            count = count + 1
            f.write("\n")
            
print(p)
f.write(str(count))
f.close()

