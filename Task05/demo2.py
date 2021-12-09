from pylab import *
from mpl_toolkits.mplot3d import Axes3D

fig = figure()
ax = Axes3D(fig)
X = np.linspace(-2.9,12,50)
Y = np.linspace(4.2,5.7,50)
x1, x2 = np.meshgrid(X, Y)
Z = 21.5 + x1 * np.sin(4*np.pi*x1) + x2 * np.sin(20*np.pi*x2)

ax.plot_surface(x1, x2, Z, rstride=1, cstride=1, cmap='hot')

show()