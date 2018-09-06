# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, world, fem, femsolver
from gridlod.world import World
import psi_functions
from visualization_tools import drawCoefficient, drawCoefficient_origin, d3plotter, d3solextra, d3sol
import buildcoef2d

fine = 16
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)
# list of coarse meshes
NList = [16]

#perturbation
alpha = 3./8.
psi = psi_functions.plateau_2d_on_one_axis(NFine, alpha)

bg = 0.01 		#background
val = 1			#values

CoefClass = buildcoef2d.Coefficient2d(NFine,
                        bg	    	        = bg,
                        val		            = val,
                        length              = 4,
                        thick               = 4,
                        space               = 4,
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

aFine = CoefClass.BuildCoefficient() # .flatten()  #  <-- we always want to have flatten form
aFine_flatten = aFine.flatten()

xpCoarse = util.pCoordinates(NFine)
xtCoarse = util.tCoordinates(NFine)

aPert = psi.inverse_transformation(aFine_flatten, xtCoarse)
aBack = psi.transformation(aPert, xtCoarse)

plt.figure("Coefficient")
drawCoefficient_origin(NFine, aFine_flatten)

plt.figure("a_perturbed")
drawCoefficient_origin(NFine, aPert)

#plt.figure("a_back")
#drawCoefficient_origin(NFine, aBack)

# jAj is the perturbed reference coefficient
jAj = psi.apply_transformation_to_bilinear_form(aFine_flatten, xtCoarse)

# TODO: Visualization for matrix valued coefficients

f = np.ones(np.prod(NFine+1))
f_pert = psi.apply_transformation_to_linear_functional(f, xpCoarse)

#d3sol(NFine,f, 'right hand side NT')
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

    uFineFull_transformed = psi.inverse_transformation(uFineFullJAJ, xpCoarse)

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