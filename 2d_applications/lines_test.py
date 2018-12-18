# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)



import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse

from gridlod import util, world, fem, femsolver, func, interp, pg, coef
from gridlod.world import World

import psi_functions
import discrete_mapping
from visualization_tools import drawCoefficient, drawCoefficient_origin, d3plotter, d3solextra, d3sol
import buildcoef2d


fine = 128
NFine = np.array([fine,fine])
#NCoeff = np.array([32,32])
NpFine = np.prod(NFine + 1)
# list of coarse meshes
N = 16
Np = 15
Nmapping = np.array([Np,Np])

xpFine = util.pCoordinates(NFine)
xtFine = util.tCoordinates(NFine)

# cq1 = np.array([0, 0.5, 0.5, 0,
#                 0, 0.5, 0.5, 0,
#                 0, 0.5, 0.5, 0,
#                 0, 0.5, 0.5, 0])
#
# cq1 = np.array([0, 0, 0.5, 0.5, 0, 0,
#                 0, 0, 0.5, 0.5, 0, 0,
#                 0, 0, 0.5, 0.5, 0, 0,
#                 0, 0, 0.5, 0.5, 0, 0,
#                 0, 0, 0.5, 0.5, 0, 0,
#                 0, 0, 0.5, 0.5, 0, 0])

cq1 = np.zeros((Np+1,Np+1))
cq1[:,int(np.round(Np/2.+0.8)-1)] = 0.5
cq1[:,int(np.round(Np/2.+0.8))] = 0.5
cq1 = cq1.flatten()

alpha = 1./8.

for_mapping = np.stack((xpFine[:,0] + alpha * func.evaluateCQ1(Nmapping, cq1, xpFine), xpFine[:,1]), axis = 1)
psi = discrete_mapping.MappingCQ1(NFine, for_mapping)

# psi_h1 = xpFine[:,0] + (1 - xpFine[:,0]) * 0.1 *xpFine[:,0]
# psi_h2 = xpFine[:,1]
# psi_h = np.stack((psi_h1, psi_h2), axis = 1)
# psi = discrete_mapping.MappingCQ1(NFine, psi_h)

# Discrete mapping
e = np.exp(1)
mapping = (np.exp(xpFine)-1)/(e-1)
#psi = discrete_mapping.MappingCQ1(NFine, mapping)

# Compute grid points and mapped grid points
# Grid naming:
# ._pert   is the grid mapped from reference to perturbed domain
# ._ref    is the grid mapped from perturbed to reference domain
xpFine_pert = psi.evaluate(xpFine)
xpFine_ref = psi.inverse_evaluate(xpFine)

xtFine_pert = psi.evaluate(xtFine)
xtFine_ref = psi.inverse_evaluate(xtFine)

bg = 0.01 		#background
val = 1			#values

CoefClass = buildcoef2d.Coefficient2d(NFine,
                        bg                  = bg,
                        val                 = val,
                        length              = 1,
                        thick               = 1,
                        space               = 16,
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
                        ChannelVertical     = True,
                        BoundarySpace       = True)

# Set reference coefficient
aCoarse_ref_shaped = CoefClass.BuildCoefficient()

aCoarse_ref = aCoarse_ref_shaped.flatten()
aFine_ref = aCoarse_ref
# aFine_ref = func.evaluateDQ0(NCoeff, aCoarse_ref, xtFine_ref)

# Compute perturbed coefficient
# Coefficient and right hand side naming:
# ._pert    is a function defined on the uniform grid in the perturbed domain
# ._ref     is a function defined on the uniform grid in the reference domain
# ._trans   is a function defined on the uniform grid on the reference domain,
#           after transformation from the perturbed domain
aFine_pert = func.evaluateDQ0(NFine, aFine_ref, xtFine_ref)
aBack_ref = func.evaluateDQ0(NFine, aFine_pert, xtFine_pert)

plt.figure("Coefficient")
drawCoefficient_origin(NFine, aFine_ref)

plt.figure("a_perturbed")
drawCoefficient_origin(NFine, aFine_pert)

plt.figure("a_back")
drawCoefficient_origin(NFine, aBack_ref)

# aFine_trans is the transformed perturbed reference coefficient
aFine_trans = np.einsum('tji, t, tkj, t -> tik', psi.Jinv(xtFine), aFine_ref, psi.Jinv(xtFine), psi.detJ(xtFine))

f_pert = np.ones(np.prod(NFine+1))
f_ref = func.evaluateCQ1(NFine, f_pert, xpFine_pert)
f_trans = np.einsum('t, t -> t', f_ref, psi.detJ(xpFine))

#d3sol(NFine,f, 'right hand side NT')
d3sol(NFine, f_trans, 'right hand side T')


NWorldCoarse = np.array([N, N])
boundaryConditions = np.array([[0, 0],[0, 0]])

NCoarseElement = NFine / NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

# Naming of solutions
# ._pert        is a solution in the perturbed domain
# ._trans       is a solution in the reference domain, after transformation
# ._trans_pert  is a solution in the perturbed domain, solved in the reference domain after transformation,
#               but then remapped to the perturbed domain
uFineFull_pert, AFine_pert, _ = femsolver.solveFine(world, aFine_pert, f_pert, None, boundaryConditions)
uFineFull_trans, AFine_trans, _ = femsolver.solveFine(world, aFine_trans, f_trans, None, boundaryConditions)
uFineFull_trans_ref, AFine_trans_ref, _ = femsolver.solveFine(world, aFine_ref, f_trans, None, boundaryConditions)
uFineFull_ref, AFine_ref, _ = femsolver.solveFine(world, aFine_ref, f_ref, None, boundaryConditions)

uFineFull_trans_pert = func.evaluateCQ1(NFine, uFineFull_trans, xpFine_ref)
uFineFull_trans_pert_ref = func.evaluateCQ1(NFine, uFineFull_trans_ref, xpFine_ref)
uFineFull_ref_pert = func.evaluateCQ1(NFine, uFineFull_ref, xpFine_ref)

#exact transformation
energy_norm = np.sqrt(np.dot(uFineFull_pert, AFine_pert * uFineFull_pert))
energy_error = np.sqrt(np.dot((uFineFull_trans_pert - uFineFull_pert), AFine_pert * (uFineFull_trans_pert - uFineFull_pert)))
print("Energy norm {}, error {}, rel. error {}".format(energy_norm, energy_error, energy_error/energy_norm))

#reference transformation with transformed f
energy_norm = np.sqrt(np.dot(uFineFull_pert, AFine_pert * uFineFull_pert))
energy_error = np.sqrt(np.dot((uFineFull_trans_pert_ref - uFineFull_pert), AFine_pert * (uFineFull_trans_pert_ref - uFineFull_pert)))
print("Energy norm {}, error {}, rel. error {}".format(energy_norm, energy_error, energy_error/energy_norm))

#pure reference transformation
energy_norm = np.sqrt(np.dot(uFineFull_pert, AFine_pert * uFineFull_pert))
energy_error = np.sqrt(np.dot((uFineFull_ref_pert - uFineFull_pert), AFine_pert * (uFineFull_ref_pert - uFineFull_pert)))
print("Energy norm {}, error {}, rel. error {}".format(energy_norm, energy_error, energy_error/energy_norm))


'''
Plot solutions
'''
fig = plt.figure(str(N))
#fig.subplots_adjust(left=0.01,bottom=0.04,right=0.99,top=0.95,wspace=0,hspace=0.2)

#ax = fig.add_subplot(221, projection='3d')
ax = fig.add_subplot(221)
ax.set_title('Solution to perturbed problem (perturbed domain)',fontsize=6)
#d3solextra(NFine, uFineFull_pert, fig, ax, min, max)
ax.imshow(np.reshape(uFineFull_pert, NFine+1), origin='lower_left')

#ax = fig.add_subplot(222, projection='3d')
ax = fig.add_subplot(222)
ax.set_title('Solution to transformed problem (reference domain)',fontsize=6)
#d3solextra(NFine, uFineFull_trans, fig, ax, min, max)
ax.imshow(np.reshape(uFineFull_trans, NFine+1), origin='lower_left')

#ax = fig.add_subplot(223, projection='3d')
ax = fig.add_subplot(223)
ax.set_title('Solution to remapped transformed problem (perturbed domain)',fontsize=6)
#d3solextra(NFine, uFineFull_trans_pert, fig, ax, min, max)
ax.imshow(np.reshape(uFineFull_trans_pert, NFine+1), origin='lower_left')

#ax = fig.add_subplot(224, projection='3d')
ax = fig.add_subplot(224)
ax.set_title('Absolute error between perturbed and remapped transformed',fontsize=6)
#d3solextra(NFine, uFineFull_trans_pert-uFineFull_pert, fig, ax, min, max)
im = ax.imshow(np.reshape(uFineFull_trans_pert - uFineFull_pert, NFine+1), origin='lower_left')
fig.colorbar(im, ax = ax)

# here, we compare the solutions.
# todo: we need a better error comparison !! This is not looking good at all.
#plt.figure('error')
#plt.loglog(NList,energy_error,'o-', basex=2, basey=2)
#plt.legend(frameon=False, fontsize="small")


plt.show()

# PGLOD this is currently failing (see todo note)

# IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, boundaryConditions)
# a_ref_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aFine_ref)
# a_trans_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aFine_trans)
#
# pglod = pg.PetrovGalerkinLOD(world, 2, IPatchGenerator, 0, 3)
# pglod.updateCorrectors(a_ref_coef,clearFineQuantities=False)
# pglod.updateCorrectors(a_trans_coef,clearFineQuantities=False)   #todo: this required error indicator with matrix valued A
#
# KFull = pglod.assembleMsStiffnessMatrix()
# MFull = fem.assemblePatchMatrix(NWorldCoarse, world.MLocCoarse)
# free = util.interiorpIndexMap(NWorldCoarse)
#
# bFull = MFull * f_trans
# KFree = KFull[free][:, free]
# bFree = bFull[free]
#
# xFree = sparse.linalg.spsolve(KFree, bFree)
# basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
#
# basisCorrectors = pglod.assembleBasisCorrectors()
#
# modifiedBasis = basis - basisCorrectors
#
# NpCoarse = np.prod(NWorldCoarse+1)
# xFull = np.zeros(NpCoarse)
# xFull[free] = xFree
# uCoarse = xFull
# uLodFine = modifiedBasis * xFull