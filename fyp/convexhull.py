import numpy as np
from scipy.optimize import *
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')

def entropy_x(distribution_x):
    # Calculate the entropy of the distribution
    entropy = -np.nansum(distribution_x * np.log2(distribution_x))
    return entropy
H = entropy_x

def obj(pxy):
    pxyz = joint(pxy)
    px = np.sum(pxyz, (1, 2))
    py = np.sum(pxyz, (0, 2))
    pz = np.sum(pxyz, (0, 1))
    return H(px) + H(py) - H(pxy) - H(pxy) + H(pz)

