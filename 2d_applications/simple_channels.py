# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, femsolver, func, interp, coef, fem, lod, pglod
from gridlod.world import World, Patch

from MasterthesisLOD import buildcoef2d
from gridlod_on_perturbations import discrete_mapping
from gridlod_on_perturbations.visualization_tools import d3sol
from MasterthesisLOD.visualize import drawCoefficientGrid, drawCoefficient


fine = 16
N = 8
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)

space = 4
thick = 2

bg = 0.01 		#background
val = 1			#values

CoefClass = buildcoef2d.Coefficient2d(NFine,
                        bg                  = bg,
                        val                 = val,
                        length              = 1,
                        thick               = thick,
                        space               = space,
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


# Discrete mapping
Nmapping = np.array([int(fine),int(fine)])
cq1 = np.zeros((int(fine)+1,int(fine)+1))

size_of_an_element = 1./fine
walk_with_perturbation =  size_of_an_element
number_of_channels = len(CoefClass.ShapeRemember)

channels_position_from_zero = space
channels_end_from_zero = channels_position_from_zero + thick

position = 7
left = 3
right = 14
cq1[:, left:right] = walk_with_perturbation
cq1[:, left - 1] = walk_with_perturbation * 2/ 3
cq1[:, left - 2] = walk_with_perturbation *1 / 3
cq1[:, right ] = walk_with_perturbation* 2/ 3
cq1[:, right +1 ] = walk_with_perturbation * 1/ 3


plt.plot(np.arange(0,fine+1),cq1[0,:], label= '$id(x) - \psi(x)$')
plt.title('Domain mapping')
plt.legend()
cq1 = cq1.flatten()

xpFine = util.pCoordinates(NFine)
xtFine = util.tCoordinates(NFine)

alpha = 1.

for_mapping = np.stack((xpFine[:,0] + alpha * func.evaluateCQ1(Nmapping, cq1, xpFine), xpFine[:,1]), axis = 1)
psi = discrete_mapping.MappingCQ1(NFine, for_mapping)


# Compute grid points and mapped grid points
# Grid naming:
# ._pert   is the grid mapped from reference to perturbed domain
# ._ref    is the grid mapped from perturbed to reference domain
xpFine_pert = psi.evaluate(xpFine)
xpFine_ref = psi.inverse_evaluate(xpFine)

xtFine_pert = psi.evaluate(xtFine)
xtFine_ref = psi.inverse_evaluate(xtFine)


# Compute perturbed coefficient
# Coefficient and right hand side naming:
# ._pert    is a function defined on the uniform grid in the perturbed domain
# ._ref     is a function defined on the uniform grid in the reference domain
# ._trans   is a function defined on the uniform grid on the reference domain,
#           after transformation from the perturbed domain
aFine_pert = func.evaluateDQ0(NFine, aFine_ref, xtFine_ref)
aBack_ref = func.evaluateDQ0(NFine, aFine_pert, xtFine_pert)

plt.figure("Coefficient")
drawCoefficient(NFine, aFine_ref)

plt.figure("a_perturbed")
drawCoefficient(NFine, aFine_pert)

plt.figure("a_back")
drawCoefficient(NFine, aBack_ref)

# aFine_trans is the transformed perturbed reference coefficient
aFine_trans = np.einsum('tji, t, tkj, t -> tik', psi.Jinv(xtFine), aFine_ref, psi.Jinv(xtFine), psi.detJ(xtFine))

f_pert = np.ones(np.prod(NFine+1))
f_ref = func.evaluateCQ1(NFine, f_pert, xpFine_pert)
f_trans = np.einsum('t, t -> t', f_ref, psi.detJ(xpFine))

#d3sol(NFine,f, 'right hand side NT')
d3sol(NFine, f_trans, 'right hand side T')

NWorldCoarse = np.array([N, N])
boundaryConditions = np.array([[0, 0],[0, 0]])

NCoarseElement = NFine // NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

# Naming of solutions
# ._pert        is a solution in the perturbed domain
# ._trans       is a solution in the reference domain, after transformation
# ._trans_pert  is a solution in the perturbed domain, solved in the reference domain after transformation,
#               but then remapped to the perturbed domain
uFineFull_pert, AFine_pert, _ = femsolver.solveFine(world, aFine_pert, f_pert, None, boundaryConditions)
uFineFull_trans, AFine_trans, _ = femsolver.solveFine(world, aFine_trans, f_trans, None, boundaryConditions)

uFineFull_trans_pert = func.evaluateCQ1(NFine, uFineFull_trans, xpFine_ref)

energy_norm = np.sqrt(np.dot(uFineFull_pert, AFine_pert * uFineFull_pert))
energy_error = np.sqrt(np.dot((uFineFull_trans_pert - uFineFull_pert), AFine_pert * (uFineFull_trans_pert - uFineFull_pert)))
print("Energy norm {}, error {}, rel. error {}".format(energy_norm, energy_error, energy_error/energy_norm))


'''
Plot solutions
'''
fig = plt.figure(str(N))
#fig.subplots_adjust(left=0.01,bottom=0.04,right=0.99,top=0.95,wspace=0,hspace=0.2)

ax = fig.add_subplot(221)
ax.set_title('Solution to perturbed problem (perturbed domain)',fontsize=6)
ax.imshow(np.reshape(uFineFull_pert, NFine+1), origin='lower_left')

ax = fig.add_subplot(222)
ax.set_title('Solution to transformed problem (reference domain)',fontsize=6)
ax.imshow(np.reshape(uFineFull_trans, NFine+1), origin='lower_left')

ax = fig.add_subplot(223)
ax.set_title('Solution to remapped transformed problem (perturbed domain)',fontsize=6)
ax.imshow(np.reshape(uFineFull_trans_pert, NFine+1), origin='lower_left')

ax = fig.add_subplot(224)
ax.set_title('Absolute error between perturbed and remapped transformed',fontsize=6)
im = ax.imshow(np.reshape(uFineFull_trans_pert - uFineFull_pert, NFine+1), origin='lower_left')
fig.colorbar(im)

k = 3

Aeye = np.tile(np.eye(2), [np.prod(NFine), 1, 1])
aFine_ref = np.einsum('tji, t-> tji', Aeye, aFine_ref)

def computeKmsij(TInd):
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    aPatch = lambda: coef.localizeCoefficient(patch, aFine_ref)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

def computeIndicators(TInd):
    aPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine_ref)
    rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine_trans)

    epsFine = lod.computeBasisErrorIndicatorFine(patchT[TInd], correctorsListT[TInd], aPatch, rPatch)
    epsCoarse = 0
    #epsCoarse = lod.computeErrorIndicatorCoarseFromCoefficients(patchT[TInd], csiT[TInd].muTPrime,  aPatch, rPatch)
    return epsFine, epsCoarse

def UpdateCorrectors(TInd):
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine_trans)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, rPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, rPatch)
    return patch, correctorsList, csi.Kmsij, csi


# Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
patchT, correctorsListT, KmsijT, csiT = zip(*map(computeKmsij, range(world.NtCoarse)))

print('compute error indicators')
epsFine, epsCoarse = zip(*map(computeIndicators, range(world.NtCoarse)))

fig = plt.figure("error indicator")
ax = fig.add_subplot(1,1,1)
np_eps = np.einsum('i,i -> i', np.ones(np.size(epsFine)), epsFine)
drawCoefficientGrid(NWorldCoarse, np_eps,fig,ax, original_style=True)

print('apply tolerance')
Elements_to_be_updated = []
for i in range(world.NtCoarse):
    if epsFine[i] >= 0.3:
        Elements_to_be_updated.append(i)
print('.... to be updated: {}'.format(np.size(Elements_to_be_updated)/np.size(epsFine)))

print('update correctors')
patchT_irrelevant, correctorsListTNew, KmsijTNew, csiTNew = zip(*map(UpdateCorrectors, Elements_to_be_updated))

print('replace Kmsij and update correctorsListT')
KmsijT_list = list(KmsijT)
correctorsListT_list = list(correctorsListT)
i=0
for T in Elements_to_be_updated:
    KmsijT_list[T] = KmsijTNew[i]
    correctorsListT_list[T] = correctorsListTNew[i]
    i+=1

KmsijT = tuple(KmsijT_list)
correctorsListT = tuple(correctorsListT_list)

print('solve the system')
KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)

MFull = fem.assemblePatchMatrix(NFine, world.MLocFine)

basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
modifiedBasis = basis - basisCorrectors

bFull = MFull * f_trans
bFull = basis.T * bFull

uFull, _ = pglod.solve(world, KFull, bFull, boundaryConditions)

uLodFine = modifiedBasis * uFull

fig = plt.figure('new figure')
ax = fig.add_subplot(121)
ax.set_title('PGLOD Solution to transformed problem (reference domain)',fontsize=6)
im = ax.imshow(np.reshape(uLodFine, NFine+1), origin='lower_left')
fig.colorbar(im)
ax = fig.add_subplot(122)
ax.set_title('FEM Solution to transformed problem (reference domain)',fontsize=6)
im = ax.imshow(np.reshape(uFineFull_trans, NFine+1), origin='lower_left')
fig.colorbar(im)

newErrorFine = np.sqrt(np.dot(uLodFine - uFineFull_trans, AFine_trans * (uLodFine - uFineFull_trans)))

print('Error: {}'.format(newErrorFine))

print('finished')

plt.show()