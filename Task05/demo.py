import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a,b,c,d = 1,2,3,4

x = np.linspace(-2.9,12,50)
y = np.linspace(4.2,5.7,50)

x1,x2 = np.meshgrid(x,y)
Z = 21.5 + x1 * np.sin(4*np.pi*x1) + x2 * np.sin(20*np.pi*x2)

fig = plt.figure()
ax = plt.axes(projection='3d')

surf = ax.plot_surface(x1, x2, Z, cmap='hot')
plt.show()