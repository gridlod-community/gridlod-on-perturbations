# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, world, fem, femsolver
from gridlod.world import World
import psi_functions
from visualization_tools import drawCoefficient, d3plotter, d3solextra, d3sol
import buildcoef2d

fine = 8
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)
# list of coarse meshes
NList = [4,8]

#perturbation
alpha = 1./4.
psi = psi_functions.plateau_2d_on_one_axis(alpha)

bg = 0.01 		#background
val = 1			#values

CoefClass = buildcoef2d.Coefficient2d(NFine,
                        bg	    	        = bg,
                        val		            = val,
                        length              = 2,
                        thick               = 2,
                        space               = 2,
                        probfactor          = 1,
                        right               = 1,
                        down                = 0,
                        diagr1              = 0,
                        diagr2              = 0,
                        diagl1              = 0,
                        diagl2              = 0,
                        LenSwitch           = None,
                        thickSwitch         = None,
                        equidistant         = True,
                        ChannelHorizontal   = None,
                        ChannelVertical     = None,
                        BoundarySpace       = True)

aFine = CoefClass.BuildCoefficient()

plt.figure("Coefficient")
drawCoefficient(NFine,aFine)

xpCoarse = util.pCoordinates(NFine)
xtCoarse = util.tCoordinates(NFine)

a_transformed = np.copy(aFine)
alpha = 1./4.

for k in range(0, np.shape(xtCoarse)[0]):
    transformed_x = psi.inverse_evaluate(xtCoarse[k])
    a_transformed[int(xtCoarse[k, 0] * fine), int(xtCoarse[k, 1] * fine)] = aFine[
        int(transformed_x[0] * fine), int(transformed_x[1] * fine)]

aPert = a_transformed

plt.figure("a_perturbed")
drawCoefficient(NFine, aPert)

# jAj is the perturbed reference coefficient
jAj_reshape = np.tile(np.eye(2), [fine,fine,1,1])
for k in range(0, np.shape(xtCoarse)[0]):
    jAj_reshape[int(xtCoarse[k, 0] * fine), int(xtCoarse[k, 1] * fine)] *= aFine[int(xtCoarse[k, 0] * fine), int(xtCoarse[k, 1] * fine)]

for k in range(0, np.shape(xtCoarse)[0]):
    Jinv = psi.Jinv(xtCoarse[k])
    jAj_reshape[int(xtCoarse[k, 0] * fine), int(xtCoarse[k, 1] * fine)] *= psi.detJ(xtCoarse[k]) * Jinv * np.transpose(Jinv)

jAj = jAj_reshape.reshape(fine*fine,2,2)
#print jAj
# TODO: Visualization for matrix valued coefficients

f = np.ones(np.prod(NFine+1))
f_pert = np.copy(f)
for k in range(0, np.shape(xpCoarse)[0]):
    f_pert[k] *= psi.detJ(xpCoarse[k])

d3sol(NFine,f, 'right hand side NT')
d3sol(NFine,f_pert, 'right hand side T')

exact_problem = []
transformed_problem = []
non_transformed_problem = []
energy_error = []
x = []
for N in NList:
    NWorldCoarse = np.array([N, N])
    boundaryConditions = np.array([[0, 0],[0, 0]])

    NCoarseElement = NFine / NWorldCoarse
    world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
 
    # grid nodes
    xpCoarse = util.pCoordinates(NFine)
    x.append(xpCoarse)
    NpCoarse = np.prod(NWorldCoarse + 1)

    uFineFull, AFine, nothing = femsolver.solveFine(world, aPert.flatten(), f, None, boundaryConditions)
    uFineFullJAJ, jAjFine, nothing = femsolver.solveFine(world, jAj, f_pert, None, boundaryConditions)

    uFineFullJAJ_reshaped = uFineFullJAJ.reshape(NFine+1)
    uFineFull_transformed_reshaped = np.copy(uFineFullJAJ_reshaped)


    for k in range(0, np.shape(xpCoarse)[0]):
        transformed_x = psi.inverse_evaluate(xpCoarse[k])
        uFineFull_transformed_reshaped[int(xpCoarse[k, 0] * fine), int(xpCoarse[k, 1] * fine)] = uFineFullJAJ_reshaped[
            int(transformed_x[0] * fine), int(transformed_x[1] * fine)]

    uFineFull_transformed = uFineFull_transformed_reshaped.flatten()
    energy_error.append(
        np.sqrt(np.dot(uFineFull - uFineFull_transformed, AFine * (uFineFull - uFineFull_transformed))))
    exact_problem.append(uFineFull)
    non_transformed_problem.append(uFineFullJAJ)
    transformed_problem.append(uFineFull_transformed)

    '''
    Plot solutions
    '''
    ymin = np.min(uFineFull)
    ymax = np.max(uFineFull)

    fig = plt.figure(str(N))
    fig.subplots_adjust(left=0.01,bottom=0.04,right=0.99,top=0.95,wspace=0,hspace=0.2)
    ax = fig.add_subplot(221, projection='3d')
    ax.set_title('exact',fontsize=16)
    d3solextra(NFine, uFineFull, fig, ax, min, max)
    ax = fig.add_subplot(222, projection='3d')
    ax.set_title('non_transformed',fontsize=16)
    d3solextra(NFine, uFineFullJAJ, fig, ax, min, max)
    ax = fig.add_subplot(223, projection='3d')
    ax.set_title('transformed',fontsize=16)
    d3solextra(NFine, uFineFull_transformed, fig, ax, min, max)
    ax = fig.add_subplot(224, projection='3d')
    ax.set_title('absolute error',fontsize=16)
    d3solextra(NFine, uFineFull_transformed-uFineFull, fig, ax, min, max)


# here, we compare the solutions.
# todo: we need a better error comparison !! This is not looking good at all.
#plt.figure('error')
#plt.loglog(NList,energy_error,'o-', basex=2, basey=2)
#plt.legend(frameon=False, fontsize="small")


plt.show()