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

import perturbations
import algorithms
from gridlod_on_perturbations.data import safe_data

ROOT = '../../2d_applications/data/test_from_scratch/'

# Set global variables for the computation

power = 8
fine = 2**power
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)

# This factor enables that A does not change for finer fine mesh
factor = 2**(power - 8)

# Coarse mesh and localization Parameter
N = 2**3
k = 3
NWorldCoarse = np.array([N, N])

# boundary Conditions
boundaryConditions = np.array([[0, 0], [0, 0]])

NCoarseElement = NFine // NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

'''
Construct right hand side
'''

f_ref = np.ones(NpFine) * 0.001
f_ref_reshaped = f_ref.reshape(NFine+1)
# f_ref_reshaped[int(0*fine/8):int(2*fine/8),int(0*fine/8):int(2*fine/8)] = 1
# f_ref_reshaped[int(6*fine/8):int(8*fine/8),int(6*fine/8):int(8*fine/8)] = 1
f_ref_reshaped[int(2*fine/8):int(6*fine/8),int(2*fine/8):int(6*fine/8)] = 1
f_ref = f_ref_reshaped.reshape(NpFine)

'''
Construct diffusion coefficient
'''

space = int(6 * factor)
thick = int(3 * factor)
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
                        ChannelVertical     = None,  # SET THIS TO TRUE TO GET CHANNELS
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
offset = squares_on_ones_axis * (number_on_one_axis-1) + number_on_one_axis + 1

numbers = [i+k + i * (number_on_one_axis*3-1) + offset for k in range(number_on_one_axis) for i in range(number_on_one_axis)]

# either Disappearance or Change in Value    ATTENTION : In this case TRANS means PERTURBED
aFine_trans = CoefClass.SpecificVanish(Number = numbers).flatten()
# aFine_trans = CoefClass.SpecificValueChange(Number = numbers, ratio = 100.).flatten()

'''
Plot diffusion coefficient and right hand side
'''

plt.figure("Coefficient")
drawCoefficient_origin(NFine, aFine_ref)

plt.figure('Perturbation')
drawCoefficient_origin(NFine, aFine_trans)

plt.figure('Right hand side')
drawCoefficient_origin(NFine+1, f_ref)

plt.show()

'''
Set the coefficient that we want to approximate and the tolerance
'''

a_Fine_to_be_approximated = aFine_trans

'''
Compute FEM
'''
uFineFull_ref, AFine_ref, _ = femsolver.solveFine(world, a_Fine_to_be_approximated, f_ref, None, boundaryConditions)

'''
Compute PGLOD 
'''

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



print('precomputing ....')

# Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
patchT, correctorsListT, KmsijT, csiT = zip(*map(computeKmsij, range(world.NtCoarse)))
patchT, correctorRhsT, RmsiT = zip(*map(computeRmsi, range(world.NtCoarse)))

Rf = pglod.assemblePatchFunction(world, patchT, correctorRhsT)
RFull = pglod.assemblePatchFunction(world, patchT, RmsiT)
MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)

print('computing error indicators')
epsCoarse = list(map(computeIndicators, range(world.NtCoarse)))

'''
Plot error indicator
'''
fig = plt.figure("error indicator")
ax = fig.add_subplot(1, 1, 1)
np_eps = np.einsum('i,i -> i', np.ones(np.size(epsCoarse)), epsCoarse)
drawCoefficientGrid(NWorldCoarse, np_eps, fig, ax, original_style=True, Gridsize=N)

AdaptiveAlgorithm = algorithms.AdaptiveAlgorithm(world = world,
                                                 k = k ,
                                                 boundaryConditions = boundaryConditions,
                                                 a_Fine_to_be_approximated = a_Fine_to_be_approximated,
                                                 aFine_ref = aFine_ref,
                                                 f_trans = f_ref,
                                                 epsCoarse = epsCoarse,
                                                 KmsijT = KmsijT,
                                                 correctorsListT = correctorsListT,
                                                 patchT = patchT,
                                                 RFull=RFull,
                                                 Rf=Rf,
                                                 MFull=MFull,
                                                 uFineFull_trans = uFineFull_ref,
                                                 AFine_trans = AFine_ref,
                                                 StartingTolerance= 100)

to_be_updatedT, energy_errorT, tmp_errorT, TOLt, uFineFull_trans_LOD = AdaptiveAlgorithm.StartAlgorithm()

# safe_data(ROOT, k, N, epsCoarse, to_be_updatedT, energy_errorT, tmp_errorT, TOLt, uFineFull_ref, uFineFull_trans_LOD)

'''
Plot solutions
'''
fig = plt.figure('Solutions')
ax = fig.add_subplot(121)
ax.set_title('Fem Solution to transformed problem',fontsize=6)
ax.imshow(np.reshape(uFineFull_ref, NFine+1), origin='lower_left')

ax = fig.add_subplot(122)
ax.set_title('PGLOD Solution to transformed problem',fontsize=6)
ax.imshow(np.reshape(uFineFull_trans_LOD, NFine+1), origin='lower_left')

'''
Errors
'''
energy_norm = np.sqrt(np.dot(uFineFull_trans_LOD, AFine_ref * uFineFull_trans_LOD))
energy_error = np.sqrt(np.dot((uFineFull_ref - uFineFull_trans_LOD), AFine_ref * (uFineFull_ref - uFineFull_trans_LOD)))
print("Energy norm {}, error {}, rel. error {}".format(energy_norm, energy_error, energy_error/energy_norm))

plt.show()