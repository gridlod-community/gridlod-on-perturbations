# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, world, fem, femsolver, func
from gridlod.world import World

import psi_functions
from visualization_tools import drawCoefficient, drawCoefficient_origin, d3plotter, d3solextra, d3sol
import buildcoef2d

fine = 128
NFine = np.array([fine,fine])
NCoeff = np.array([16,16])
NpFine = np.prod(NFine + 1)
# list of coarse meshes
N = 16

#perturbation
alpha = 3./8.
psi = psi_functions.plateau_2d_on_one_axis(NFine, alpha)

# Compute grid points and mapped grid points
# Grid naming:
# ._phys   is the grid mapped from reference to physical domain
# ._ref    is the grid mapped from physical to reference domain
xpFine = util.pCoordinates(NFine)
xpFine_phys = psi.evaluate(xpFine)
xpFine_ref = psi.inverse_evaluate(xpFine)

xtFine = util.tCoordinates(NFine)
xtFine_phys = psi.evaluate(xtFine)
xtFine_ref = psi.inverse_evaluate(xtFine)

bg = 0.01 		#background
val = 1			#values

CoefClass = buildcoef2d.Coefficient2d(NCoeff,
                        bg                  = bg,
                        val                 = val,
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

# Set reference coefficient
aCoarse_ref_shaped = CoefClass.BuildCoefficient()
aCoarse_ref = aCoarse_ref_shaped.flatten()
aFine_ref = func.evaluateDQ0(NCoeff, aCoarse_ref, xtFine_ref)

# Compute physical coefficient
# Coefficient and right hand side naming:
# ._phys    is a function defined on the uniform grid in the physical domain
# ._ref     is a function defined on the uniform grid in the reference domain
# ._trans   is a function defined on the uniform grid on the reference domain,
#           after transformation from the physical domain
aFine_phys = func.evaluateDQ0(NFine, aFine_ref, xtFine_ref)
aBack_ref = func.evaluateDQ0(NFine, aFine_phys, xtFine_phys)

plt.figure("Coefficient")
drawCoefficient_origin(NFine, aFine_ref)

plt.figure("a_perturbed")
drawCoefficient_origin(NFine, aFine_phys)

plt.figure("a_back")
drawCoefficient_origin(NFine, aBack_ref)

# aFine_trans is the transformed perturbed reference coefficient
aFine_trans = np.einsum('tji, t, tkj, t -> tik', psi.Jinv(xtFine), aFine_ref, psi.Jinv(xtFine), psi.detJ(xtFine))

f_phys = np.ones(np.prod(NFine+1))
f_ref = func.evaluateCQ1(NFine, f_phys, xpFine_phys)
f_trans = np.einsum('t, t -> t', f_ref, psi.detJ(xpFine))

#d3sol(NFine,f, 'right hand side NT')
d3sol(NFine, f_trans, 'right hand side T')


NWorldCoarse = np.array([N, N])
boundaryConditions = np.array([[0, 0],[0, 0]])

NCoarseElement = NFine / NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

# Naming of solutions
# ._phys        is a solution in the physical domain
# ._trans       is a solution in the reference domain, after transformation
# ._trans_phys  is a solution in the physical domain, solved in the reference domain after transformation,
#               but then remapped to the physical domain
uFineFull_phys, AFine_phys, _ = femsolver.solveFine(world, aFine_phys, f_phys, None, boundaryConditions)
uFineFull_trans, AFine_trans, _ = femsolver.solveFine(world, aFine_trans, f_trans, None, boundaryConditions)

uFineFull_trans_phys = func.evaluateCQ1(NFine, uFineFull_trans, xpFine_ref)

energy_norm = np.sqrt(np.dot(uFineFull_phys, AFine_phys * uFineFull_phys))
energy_error = np.sqrt(np.dot((uFineFull_trans_phys - uFineFull_phys), AFine_phys * (uFineFull_trans_phys - uFineFull_phys)))
print("Energy norm {}, error {}, rel. error {}".format(energy_norm, energy_error, energy_error/energy_norm))

# energy_error.append(
#     np.sqrt(np.dot(uFineFull - uFineFull_transformed, AFine * (uFineFull - uFineFull_transformed))))
# exact_problem.append(uFineFull)
# non_transformed_problem.append(uFineFullJAJ)
# transformed_problem.append(uFineFull_transformed)

'''
Plot solutions
'''
fig = plt.figure(str(N))
#fig.subplots_adjust(left=0.01,bottom=0.04,right=0.99,top=0.95,wspace=0,hspace=0.2)

#ax = fig.add_subplot(221, projection='3d')
ax = fig.add_subplot(221)
ax.set_title('Solution to physical problem (physical domain)',fontsize=16)
#d3solextra(NFine, uFineFull_phys, fig, ax, min, max)
ax.imshow(np.reshape(uFineFull_phys, NFine+1), origin='lower_left')

#ax = fig.add_subplot(222, projection='3d')
ax = fig.add_subplot(222)
ax.set_title('Solution to transformed problem (reference domain)',fontsize=16)
#d3solextra(NFine, uFineFull_trans, fig, ax, min, max)
ax.imshow(np.reshape(uFineFull_trans, NFine+1), origin='lower_left')

#ax = fig.add_subplot(223, projection='3d')
ax = fig.add_subplot(223)
ax.set_title('Solution to remapped transformed problem (physical domain)',fontsize=16)
#d3solextra(NFine, uFineFull_trans_phys, fig, ax, min, max)
ax.imshow(np.reshape(uFineFull_trans_phys, NFine+1), origin='lower_left')

#ax = fig.add_subplot(224, projection='3d')
ax = fig.add_subplot(224)
ax.set_title('Absolute error between physical and remapped transformed',fontsize=16)
#d3solextra(NFine, uFineFull_trans_phys-uFineFull_phys, fig, ax, min, max)
ax.imshow(np.reshape(uFineFull_trans_phys - uFineFull_phys, NFine+1), origin='lower_left')


# here, we compare the solutions.
# todo: we need a better error comparison !! This is not looking good at all.
#plt.figure('error')
#plt.loglog(NList,energy_error,'o-', basex=2, basey=2)
#plt.legend(frameon=False, fontsize="small")


plt.show()
