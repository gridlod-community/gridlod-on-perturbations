# This file is part of the master thesis "Variational crimes in the Localized orthogonal decomposition method":
#   https://github.com/TiKeil/Masterthesis-LOD.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm

from gridlod import util


def drawCoefficient_origin(N, a, transformed = False, lim=None):
    # This is drawCoefficient from test_pgtransport.py in gridlod
    if a.ndim == 3:
        a = np.linalg.norm(a, axis=(1, 2), ord=2)

    aCube = a.reshape(N, order='F')
    aCube = np.ascontiguousarray(aCube.T)

    plt.clf()

    cmap = plt.cm.hot_r
    if lim is not None:
        plt.imshow(aCube,
                   origin='lower_left',
                   interpolation='none', cmap=cmap,
                   norm=matplotlib.colors.LogNorm(), vmax=lim[0], vmin=lim[1])
    else:
        plt.imshow(aCube,
                   origin='lower_left',
                   interpolation='none', cmap=cmap,
                   norm=matplotlib.colors.LogNorm())

    plt.xticks([])
    plt.yticks([])
    if transformed:
        cb = plt.colorbar()
        font_size = 14  # Adjust as appropriate.
        cb.ax.tick_params(labelsize=font_size)
        cb.set_ticks([0.1,1,10])


def draw_f(N, a):
    aCube = a.reshape(N, order='F')
    aCube = np.ascontiguousarray(aCube.T)

    plt.clf()

    cmap = plt.cm.hot_r

    if np.isclose(np.linalg.norm(a),0):
        plt.imshow(aCube,
                   origin='lower_left',
                   interpolation='none', cmap=cmap)
    else:
        plt.imshow(aCube,
                   origin='lower_left',
                   interpolation='none', cmap=cmap)
    cb = plt.colorbar()
    font_size = 14  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    plt.xticks([])
    plt.yticks([])

def draw_indicator(N, a, colorbar=True, original_style = True, Gridsize = 4, string='', logplot = True):
    fig = plt.figure("error indicator {}".format(string))
    ax = fig.add_subplot(1, 1, 1)

    a = np.array([(1e-14 if a_i is 0 else a_i) for a_i in a])
    aCube = a.reshape(N, order ='F')
    aCube = np.ascontiguousarray(aCube.T)

    cmap = cm.hot_r
    cmap.set_bad('w')

    te = Gridsize
    major_ticks = np.arange(0, te, 1)
    if np.isclose(np.linalg.norm(a),0):
        if original_style:
            im = ax.imshow(aCube, cmap=cmap, origin='lower', extent=[0, te, 0, te])
        else:
            im = ax.imshow(aCube, cmap=cmap, extent=[0, te, 0, te])
    else:
        if logplot:
            if original_style:
                im = ax.imshow(aCube, cmap=cmap, origin='lower', extent=[0, te, 0, te],
                               norm=matplotlib.colors.LogNorm())
            else:
                im = ax.imshow(aCube, cmap=cmap, extent=[0, te, 0, te], norm=matplotlib.colors.LogNorm())
        else:
            if original_style:
                im = ax.imshow(aCube, cmap=cmap, origin='lower', extent=[0, te, 0, te])
            else:
                im = ax.imshow(aCube, cmap=cmap, extent=[0, te, 0, te])

    ax.axis([0, te, 0, te])
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    fig.subplots_adjust(left=0.00, bottom=0.02, right=1, top=0.95, wspace=0.2, hspace=0.2)
    ax.grid(which='both')
    ax.grid(which='major', linestyle="-", color="grey", alpha=0.4)
    if colorbar:
        cb = fig.colorbar(im)
        font_size = 16  # Adjust as appropriate.
        cb.ax.tick_params(labelsize=font_size)

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

