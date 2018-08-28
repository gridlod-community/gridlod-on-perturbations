# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, world, fem, femsolver
from gridlod.world import World
import psi_functions

# this fine resolution should be enough
fine = 512
NFine = np.array([fine])
NpFine = np.prod(NFine + 1)
# list of coarse meshes
NList = [2, 4, 8, 16, 32, 64, 128, 256]


# construction of the coefficients.
# The plot is ecactly what we want and

alpha = 1./4.
psi = psi_functions.plateau_1d(alpha)


aFine = np.ones(fine)
aFine /= 10
aPert = np.copy(aFine)

for i in range(int(fine* 2/8.) - 1, int(fine * 3/8.) - 1):
    aFine[i] = 1

for i in range(int(fine* 5/8.) - 1, int(fine * 6/8.) - 1):
    aFine[i] = 1


xpCoarse = util.pCoordinates(NFine).flatten()
xtCoarse = util.tCoordinates(NFine).flatten()

a_transformed = np.copy(aFine)
alpha = 1./4.
for k in range(0, np.shape(xtCoarse)[0]):
    transformed_x = psi.evaluate(xtCoarse[k])

    # print('point {} has been transformed to {} and the new index is {}'.format(xpCoarse[k],transformed_x,index_search(transformed_x, xpCoarse)-1))
    a_transformed[k] = aFine[int(transformed_x*NFine)]

aPert = a_transformed

# this is the mesh size
delta_h = 1. / fine

# jAj is the perturbed reference coefficient
jAj = np.copy(aFine)

for i in range(fine):
    point = xtCoarse[i]
    J = psi.J(point)
    detJ = psi.detJ(point)
    jAj[i] *= detJ/(J*J)  # NOTE: one dimensional case

# interior nodes for plotting
xt = util.tCoordinates(NFine).flatten()

# This is the right hand side and the one occuring from the transformation
f = np.ones(fine + 1)
f_pert = np.copy(f)
for i in range(0, fine+1):
    point = xpCoarse[i]
    detJ = psi.detJ(point)
    f_pert[i] *= detJ

plt.figure('right hand side')
plt.plot(np.arange(fine + 1), f, label='NT')
plt.plot(np.arange(fine + 1), f_pert, label='TR')
plt.legend()

# plot coefficients and compare them
plt.figure('Coefficient_pert')
plt.plot(xt, aFine, label='$A$')
plt.plot(xt, aPert, label='$Apert$')
plt.plot(xt, jAj, label='$JAJ$')
plt.grid(True)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=16)
plt.ylabel('$y$', fontsize=16)
plt.xlabel('$x$', fontsize=16)
plt.legend(frameon=False, fontsize=16)


exact_problem = []
transformed_problem = []
non_transformed_problem = []
energy_error = []
x = []
for N in NList:
    NWorldCoarse = np.array([N])
    boundaryConditions = np.array([[0, 0]])

    NCoarseElement = NFine / NWorldCoarse
    world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
    AFine = fem.assemblePatchMatrix(NFine, world.ALocFine, aPert)
    jAjFine = fem.assemblePatchMatrix(NFine, world.ALocFine, jAj)

    # grid nodes
    xpCoarse = util.pCoordinates(NFine).flatten()
    x.append(xpCoarse)
    NpCoarse = np.prod(NWorldCoarse + 1)
    uCoarseFull, nothing = femsolver.solveCoarse(world, aPert, f, None, boundaryConditions)
    uCoarseFullJAJ, nothing = femsolver.solveCoarse(world, jAj, f_pert, None, boundaryConditions)

    basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
    uCoarseFull = basis * uCoarseFull
    uCoarseFullJAJ = basis * uCoarseFullJAJ

    uCoarseFull_transformed = np.copy(uCoarseFullJAJ)
    k = 0

    for k in range(0, np.shape(xpCoarse)[0]):
        transformed_x = psi.evaluate(xpCoarse[k])
        uCoarseFull_transformed[k] = uCoarseFullJAJ[int(transformed_x*NFine)]

    energy_error.append(np.sqrt(np.dot(uCoarseFull - uCoarseFull_transformed, AFine * (uCoarseFull - uCoarseFull_transformed))))
    exact_problem.append(uCoarseFull)
    non_transformed_problem.append(uCoarseFullJAJ)
    transformed_problem.append(uCoarseFull_transformed)


# here, we compare the solutions.
# todo: we need a better error comparison !! This is not looking good at all.
plt.figure('error')
plt.loglog(NList,energy_error,'o-', basex=2, basey=2)
plt.legend(frameon=False, fontsize="small")

plt.figure('smallest error')
error = exact_problem[7] - transformed_problem[7]
plt.plot(xpCoarse, error)


#Plot solutions
plt.figure('FEM-Solutions', figsize=(16, 9))
plt.subplots_adjust(left=0.01, bottom=0.04, right=0.99, top=0.95, wspace=0.1, hspace=0.2)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
for i in xrange(len(NList)):
    plt.subplot(2,4,i+1)
    plt.plot(x[i], exact_problem[i], '--', label='EX')
    plt.plot(x[i], non_transformed_problem[i], '--', label='NT')
    plt.plot(x[i], transformed_problem[i], '--', label='T')
    plt.title(r'$1/H=$ ' + str(2**(i+1)), fontsize=18)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
                    labelleft=False)
    plt.legend(frameon=False, fontsize=16)

plt.show()
