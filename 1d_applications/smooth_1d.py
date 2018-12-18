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
NList = [128, 256]

alpha = 1.
psi = psi_functions.smooth_1d(NFine, alpha)

aFine = np.ones(fine)
aFine /= 10
aPert = np.copy(aFine)

for i in range(int(fine* 2/8.) - 1, int(fine * 3/8.) - 1):
    aFine[i] = 1

for i in range(int(fine* 5/8.) - 1, int(fine * 6/8.) - 1):
    aFine[i] = 1


xpCoarse = util.pCoordinates(NFine).flatten()
xtCoarse = util.tCoordinates(NFine).flatten()

aPert = psi.inverse_transformation(aFine, xtCoarse)

jAj = psi.apply_transformation_to_bilinear_form(aFine, xtCoarse)

f = np.ones(fine + 1)
f_pert = psi.apply_transformation_to_linear_functional(f, xpCoarse)

plt.figure('right hand side')
plt.plot(np.arange(fine + 1), f, label='NT')
plt.plot(np.arange(fine + 1), f_pert, label='TR')
plt.legend()

# interior nodes for plotting
xt = util.tCoordinates(NFine).flatten()

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

plt.figure('psi visualization')
ret = psi.evaluate(xpCoarse)
inverse_ret = psi.inverse_evaluate(xpCoarse)
plt.plot(xpCoarse,ret)
plt.plot(xpCoarse,xpCoarse)
plt.plot(xpCoarse,inverse_ret)

exact_problem = []
transformed_problem = []
non_transformed_problem = []
energy_error = []
x = []
for N in NList:
    NWorldCoarse = np.array([N])
    boundaryConditions = np.array([[0, 0]])

    NCoarseElement = NFine // NWorldCoarse
    world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

    # grid nodes
    xpCoarse = util.pCoordinates(NFine).flatten()
    x.append(xpCoarse)
    NpCoarse = np.prod(NWorldCoarse + 1)
    uFineFull, AFine, nothing = femsolver.solveFine(world, aPert, f, None, boundaryConditions)
    uFineFullJAJ, jAjFine, nothing = femsolver.solveFine(world, jAj, f_pert, None, boundaryConditions)

    uFineFull_transformed = np.copy(uFineFullJAJ)
    k = 0

    uFineFull_transformed = psi.inverse_transformation(uFineFullJAJ, xpCoarse)

    energy_error.append(np.sqrt(np.dot(uFineFull - uFineFull_transformed, AFine * (uFineFull - uFineFull_transformed))))
    exact_problem.append(uFineFull)
    non_transformed_problem.append(uFineFullJAJ)
    transformed_problem.append(uFineFull_transformed)


# here, we compare the solutions.
# todo: we need a better error comparison !! This is not looking good at all.
plt.figure('error')
plt.loglog(NList,energy_error,'o-', basex=2, basey=2)
plt.legend(frameon=False, fontsize="small")

plt.figure('smallest error')
error = exact_problem[1] - transformed_problem[1]
plt.plot(xpCoarse, error)


#Plot solutions
plt.figure('FEM-Solutions', figsize=(16, 9))
plt.subplots_adjust(left=0.01, bottom=0.04, right=0.99, top=0.95, wspace=0.1, hspace=0.2)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
for i in range(len(NList)):
    plt.subplot(1,2,i+1)
    plt.plot(x[i], exact_problem[i], '--', label='EX')
    plt.plot(x[i], non_transformed_problem[i], '--', label='NT')
    plt.plot(x[i], transformed_problem[i], '--', label='T')
    plt.title(r'$1/H=$ ' + str(2**(i+1)), fontsize=18)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
                    labelleft=False)
    plt.legend(frameon=False, fontsize=16)

plt.show()