# This file is part of the master thesis "Variational crimes in the Localized orthogonal decomposition method":
#   https://github.com/TiKeil/Masterthesis-LOD.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MultipleLocator
from matplotlib import cm

from gridlod import util


def drawCoefficient_origin(N, a):
    # This is drawCoefficient from test_pgtransport.py in gridlod
    if a.ndim == 3:
        a = np.linalg.norm(a, axis=(1, 2), ord=2)

    aCube = np.log10(a.reshape(N, order='F'))
    aCube = np.ascontiguousarray(aCube.T)

    plt.clf()

    cmap = plt.cm.hot_r

    plt.imshow(aCube,
               origin='lower_left',
               interpolation='none', cmap=cmap)
    plt.axis('off')

def d3sol(N, s, String='FinescaleSolution'):
    '''
    3d solution
    '''
    fig = plt.figure(String)
    ax = fig.add_subplot(111, projection='3d')

    xp = util.pCoordinates(N)
    X = xp[0:, 1:].flatten()
    Y = xp[0:, :1].flatten()
    X = np.unique(X)
    Y = np.unique(Y)

    X, Y = np.meshgrid(X, Y)

    uLodFine = s.reshape(N + 1)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, uLodFine, cmap=cm.jet)
    ymin, ymax = ax.set_zlim()
    ax.set_zticks((ymin, ymax))
    ax.set_zlabel('$z$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.zaxis.set_major_locator(LinearLocator(10))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

