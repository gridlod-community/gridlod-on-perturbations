# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, femsolver, interp, coef, fem, lod, pglod
from gridlod.world import World, Patch

from MasterthesisLOD import buildcoef2d
from gridlod_on_perturbations.visualization_tools import drawCoefficient_origin
from MasterthesisLOD.visualize import drawCoefficientGrid

from perturbations import BendingInOneArea

# Set global variables for the computation

power = 8
fine = 2**power
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)

# This factor enables that A does not change for finer fine mesh
factor = 2**(power - 8)

# Coarse mesh
N = 2**3
NWorldCoarse = np.array([N, N])

# boundary Conditions
boundaryConditions = np.array([[0, 0], [0, 1]])

NCoarseElement = NFine // NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)


'''
diffusion coefficient
'''

space = int(12 * factor)
thick = int(4 * factor)
bg = 0.1		#background
val = 1			#values

# See my master thesis Chapter 7 for detailed explanation on buildcoef2d
CoefClass = buildcoef2d.Coefficient2d(NFine,
                        bg                  = bg,
                        val                 = val,
                        length              = thick,
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
                        ChannelVertical     = True,  # SET THIS TO TRUE TO GET CHANNELS
                        BoundarySpace       = True)

# Set reference coefficient
aFine_ref = CoefClass.BuildCoefficient().flatten()

'''
Perturbation using buildcoef2d
'''

# Get information from CoefClass to know the number of squares
squares_on_ones_axis = np.sqrt(np.shape(CoefClass.ShapeRemember)[0])
number_on_one_axis = int(squares_on_ones_axis/3)

offset = 0    #starting point
offset = squares_on_ones_axis * (number_on_one_axis +1) + number_on_one_axis + 1

numbers = [i+k + i * (number_on_one_axis*3 +1) + offset for k in range(number_on_one_axis) for i in range(number_on_one_axis)]

# either Disappearance or Change in Value
aFine_pert = CoefClass.SpecificVanish(Number = numbers).flatten()
aFine_pert = CoefClass.SpecificValueChange(Number = numbers, ratio = 100.).flatten()

'''
Domain mapping perturbation
'''

# psi = create_bending_perturbation(world)
# aFine_pert = compute_perturbation(world, psi, aFine_ref)

bending_perturbation = BendingInOneArea(world, bending_factor=5)
aFine_pert = bending_perturbation.compute_perturbation(aFine_ref)

'''
Plot diffusion coefficient
'''
plt.figure("Coefficient")
drawCoefficient_origin(NFine, aFine_ref)

plt.figure("Perturbed coefficient")
drawCoefficient_origin(NFine, aFine_pert)

fig = plt.figure("Coefficient with grid")
ax = fig.add_subplot(1, 1, 1)
drawCoefficientGrid(NFine, aFine_ref, fig, ax, original_style=True, Gridsize = N)

'''
Construct right hand side
'''

f_ref = np.ones(NpFine) * 0.01
f_ref_reshaped = f_ref.reshape(NFine+1)
f_ref_reshaped[int(0*fine/8):int(2*fine/8),int(0*fine/8):int(2*fine/8)] = 1
# f_ref_reshaped[int(6*fine/8):int(8*fine/8),int(6*fine/8):int(8*fine/8)] = 1
# f_ref_reshaped[int(3*fine/8):int(5*fine/8),int(3*fine/8):int(5*fine/8)] = 1
f_ref = f_ref_reshaped.reshape(NpFine)

'''
Plot right hand side
'''
plt.figure('Right hand side')
drawCoefficient_origin(NFine+1, f_ref)

'''
Set the coefficient that we want to approximate
'''

a_Fine_to_be_approximated = aFine_ref
a_Fine_to_be_approximated = aFine_pert

tol = 100

'''
Compute FEM
'''
uFineFull_ref, AFine_ref, _ = femsolver.solveFine(world, a_Fine_to_be_approximated, f_ref, None, boundaryConditions)

'''
Compute PGLOD 
'''
k = 3

def computeKmsij(TInd):
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    aPatch = lambda: coef.localizeCoefficient(patch, aFine_ref)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

def computeRmsi(TInd):
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    aPatch = lambda: coef.localizeCoefficient(patch, a_Fine_to_be_approximated)
    MRhsList = [f_ref[util.extractElementFine(world.NWorldCoarse,
                                          world.NCoarseElement,
                                          patch.iElementWorldCoarse,
                                          extractElements=False)]];

    correctorRhs = lod.computeElementCorrector(patch, IPatch, aPatch, None, MRhsList)[0]
    Rmsi = lod.computeRhsCoarseQuantities(patch, correctorRhs, aPatch)
    return patch, correctorRhs, Rmsi

def computeIndicators(TInd):
    aPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine_ref)
    rPatch = lambda: coef.localizeCoefficient(patchT[TInd], a_Fine_to_be_approximated)

    epsCoarse = lod.computeErrorIndicatorCoarseFromCoefficients(patchT[TInd], csiT[TInd].muTPrime,  aPatch, rPatch)
    return epsCoarse

def UpdateCorrectors(TInd):
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    rPatch = lambda: coef.localizeCoefficient(patchT[TInd], a_Fine_to_be_approximated)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, rPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, rPatch)
    return patch, correctorsList, csi.Kmsij, csi

def UpdateElements(epsCoarse, tol, KmsijT, correctorsListT):
    print('apply tolerance')
    Elements_to_be_updated = []
    for i in range(world.NtCoarse):
        if epsCoarse[i] > tol:
            Elements_to_be_updated.append(i)
    print('... to be updated: {}'.format(np.size(Elements_to_be_updated) / np.size(epsCoarse)), end='', flush=True)

    if np.size(Elements_to_be_updated) != 0:
        print('... update correctors')
        patchT_irrelevant, correctorsListTNew, KmsijTNew, csiTNew = zip(*map(UpdateCorrectors, Elements_to_be_updated))

        print('replace Kmsij and update correctorsListT')
        KmsijT_list = list(KmsijT)
        correctorsListT_list = list(correctorsListT)
        i = 0
        for T in Elements_to_be_updated:
            KmsijT_list[T] = KmsijTNew[i]
            correctorsListT_list[T] = correctorsListTNew[i]
            i += 1

        KmsijT = tuple(KmsijT_list)
        correctorsListT = tuple(correctorsListT_list)

        return KmsijT, correctorsListT
    else:
        print('... there is nothing to be updated')
        return KmsijT, correctorsListT


# Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
print('compute Kmsij and Rmsi for all T')
patchT, correctorsListT, KmsijT, csiT = zip(*map(computeKmsij, range(world.NtCoarse)))
patchT, correctorRhsT, RmsiT = zip(*map(computeRmsi, range(world.NtCoarse)))

print('compute error indicators')
epsCoarse = list(map(computeIndicators, range(world.NtCoarse)))

'''
Plot error indicator
'''
fig = plt.figure("error indicator")
ax = fig.add_subplot(1, 1, 1)
np_eps = np.einsum('i,i -> i', np.ones(np.size(epsCoarse)), epsCoarse)
drawCoefficientGrid(NWorldCoarse, np_eps, fig, ax, original_style=True, Gridsize=N)


KmsijT, correctorsListT = UpdateElements(epsCoarse, tol, KmsijT, correctorsListT)

'''
Solve the PGLOD system
'''
print('assemble and solve system')
KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
RFull = pglod.assemblePatchFunction(world, patchT, RmsiT)
MFull = fem.assemblePatchMatrix(NFine, world.MLocFine)

basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
modifiedBasis = basis - basisCorrectors

bFull = basis.T * MFull * f_ref - RFull

uFull, _ = pglod.solve(world, KFull, bFull, boundaryConditions)

uLodFine = modifiedBasis * uFull
uLodFine += pglod.assemblePatchFunction(world, patchT, correctorRhsT)

'''
Plot solutions
'''
fig = plt.figure('Solutions')
ax = fig.add_subplot(121)
ax.set_title('Fem Solution to reference problem',fontsize=6)
ax.imshow(np.reshape(uFineFull_ref, NFine+1), origin='lower_left')

ax = fig.add_subplot(122)
ax.set_title('PGLOD Solution to reference problem',fontsize=6)
ax.imshow(np.reshape(uLodFine, NFine+1), origin='lower_left')

'''
Errors
'''
energy_norm = np.sqrt(np.dot(uLodFine, AFine_ref * uLodFine))
energy_error = np.sqrt(np.dot((uFineFull_ref - uLodFine), AFine_ref * (uFineFull_ref - uLodFine)))
print("Energy norm {}, error {}, rel. error {}".format(energy_norm, energy_error, energy_error/energy_norm))

plt.show()