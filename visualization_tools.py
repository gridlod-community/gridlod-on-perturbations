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
    aCube = np.log10(a.reshape(N, order='F'))
    aCube = np.ascontiguousarray(aCube.T)
    plt.clf()

    cmap = plt.cm.plasma

    plt.imshow(aCube,
               origin='lower',
               interpolation='none',
               cmap=cmap)
    plt.colorbar()


def drawCoefficient(N, a, greys=False, normalize=None):
    '''
    visualizing the 2d coefficient
    '''
    aCube = a.reshape(N)
    plt.clf()

    if greys:
        cmap = 'Greys'
    else:
        cmap = cm.plasma

    if normalize is None:
        plt.imshow(aCube,
                   origin='upper',
                   interpolation='none',
                   cmap=cmap)

    else:
        plt.imshow(aCube,
                   origin='upper',
                   interpolation='none',
                   cmap=cmap,
                   vmin=normalize[0],
                   vmax=normalize[1])

    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
                    labelleft=False)
    plt.subplots_adjust(left=0.00, bottom=0.02, right=1, top=0.95, wspace=0.2, hspace=0.2)




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

def d3solextra(N, s, fig, ax, ymin, ymax):
    '''
    function for 2d fem example
    '''

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
    ax.set_zlim(ymin, ymax)
    ax.set_zlabel('$z$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.axis('off')

def d3plotter(N, s, String='FinescaleSolution', boundary=None, zmax=None, zmin=None):
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
    if zmin is not None:
        surf = ax.plot_surface(X, Y, uLodFine, cmap=cm.coolwarm, vmin=zmin, vmax=zmax)
    else:
        surf = ax.plot_surface(X, Y, uLodFine, cmap=cm.coolwarm, label=String)

    if boundary is not None:
        surf = ax.plot_surface(X, Y, uLodFine, cmap=cm.coolwarm, vmin=boundary[0], vmax=boundary[1])
        ax.set_zlim(boundary[0], boundary[1])
    ax.zaxis.set_major_locator(LinearLocator(10))
    if zmin is not None:
        ax.set_zlim(zmin, zmax)
    ax.axis('off')
    ax.grid(False)
    fig.subplots_adjust(left=0.00, bottom=0.00, right=1, top=1, wspace=0.2, hspace=0.2)
    ax.set_title(String)


def plot_error_indicator(eps, recomputefractionsafe, NWorldCoarse, String):
    eps.sort()
    eps = np.unique(eps)
    es = []
    for i in range(0, np.size(eps)):
        es.append(eps[np.size(eps) - i - 1])
    if np.size(recomputefractionsafe) != np.size(es):
        es.append(0)
    if np.size(recomputefractionsafe) != np.size(es):
        es.append(0)
    plt.figure("Sorted error indicator for " + String)
    elemente = np.arange(np.prod(NWorldCoarse))
    plt.plot(recomputefractionsafe, es, label=String)
    plt.title("Sorted error indicator for " + String)
    plt.ylabel('$e_{u}$', fontsize=20)
    plt.xlabel('Updated correctors in %', fontsize=20)
    plt.subplots_adjust(left=0.15, bottom=0.14, right=0.99, top=0.95, wspace=0.2, hspace=0.2)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.tick_params(axis='both', which='minor', labelsize=17)
    # plt.savefig('pic/indi_'+ String + '.pdf')


def plot_error_indicator_all(eps, recomputefractionsafe, NWorldCoarse, String):
    eps.sort()
    eps = np.unique(eps)
    es = []
    for i in range(0, np.size(eps)):
        es.append(eps[np.size(eps) - i - 1])
    if np.size(recomputefractionsafe) != np.size(es):
        es.append(0)
    plt.figure("Compare sorted error indicators")
    plt.plot(recomputefractionsafe, es, label=String)
    plt.title("Comparison of sorted error indicators")
    plt.ylabel('$e_{u}$', fontsize=20)
    plt.xlabel('Updated correctors in %', fontsize=20)
    plt.legend(loc='upper right', fontsize=17)
    plt.subplots_adjust(left=0.15, bottom=0.14, right=0.99, top=0.95, wspace=0.2, hspace=0.2)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.tick_params(axis='both', which='minor', labelsize=17)
    # plt.savefig('pic/compare_indi.pdf')


def plot_VCLOD_error(errorbest, errorworst, errorplotinfo, recomputefractionsafe, String):
    size = np.size(recomputefractionsafe)
    iden = []
    for i in range(0, size):
        iden.append(errorworst[0] - ((errorworst[0] - errorbest[0]) / size) * i)

    plt.figure("Errors for " + String)
    errorplotinfo1 = []
    for l in range(0, np.size(errorplotinfo)):
        errorplotinfo1.append(errorplotinfo[l] / errorbest[l])

    plt.plot(recomputefractionsafe, errorplotinfo1)
    plt.title("Errors for " + String)
    ymin, ymax = plt.ylim()
    plt.ylabel('$e_{vc}$', fontsize=20)
    plt.xlabel('Updated correctors in %', fontsize=20)
    plt.subplots_adjust(left=0.15, bottom=0.14, right=0.99, top=0.95, wspace=0.2, hspace=0.2)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.tick_params(axis='both', which='minor', labelsize=17)
    # plt.savefig('pic/error_'+ String + '.pdf')


def plot_VCLOD_error_all(errorbest, errorworst, errorplotinfo, recomputefractionsafe, String):
    size = np.size(recomputefractionsafe)
    iden = []
    for i in range(0, size):
        iden.append(errorworst[0] - ((errorworst[0] - errorbest[0]) / size) * i)

    plt.figure("Errors for " + String)
    errorplotinfo1 = []
    for l in range(0, np.size(errorplotinfo)):
        errorplotinfo1.append(errorplotinfo[l] / errorbest[l])

    plt.figure("Compare error")
    plt.plot(recomputefractionsafe, errorplotinfo1, label=String)
    plt.title("VCLOD error comparison")
    ymin, ymax = plt.ylim()
    plt.legend(loc='upper right', fontsize=17)
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.tick_params(axis='both', which='minor', labelsize=17)
    plt.subplots_adjust(left=0.15, bottom=0.14, right=0.99, top=0.95, wspace=0.2, hspace=0.2)
    plt.ylabel('$e_{vc}/e_{pert}$', fontsize=20)
    plt.xlabel('Updated correctors in %', fontsize=20)
    plt.legend(loc='upper right', fontsize=17)
    plt.grid()
    # plt.savefig('pic/compare_errors.pdf')