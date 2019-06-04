# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib.pyplot as plt
import random

from gridlod import util, femsolver, interp, coef, fem, lod, pglod
from gridlod.world import World, Patch

from MasterthesisLOD import buildcoef2d
from gridlod_on_perturbations.visualization_tools import drawCoefficient_origin
from MasterthesisLOD.visualize import drawCoefficientGrid

import perturbations
import algorithms
from gridlod_on_perturbations.data import store_all_data

ROOT = '../../2d_applications/data/Axels_Experiment/20x20dots'

# Set global variables for the computation

potenz = 8
factor = 2**(potenz - 8)
fine = 2**potenz
N = 2**5
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)

k = 4  # goes like log H
NWorldCoarse = np.array([N, N])

# boundary Conditions
boundaryConditions = np.array([[0, 0], [0, 0]])

NCoarseElement = NFine // NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

'''
Construct diffusion coefficient
'''

space = 8 * factor
thick = 4 * factor

bg = 0.01		#background
val = 1			#values

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
                        ChannelVertical     = None,
                        BoundarySpace       = True)


# Set reference coefficient
aFine_ref_shaped = CoefClass.BuildCoefficient()
aFine_ref = aFine_ref_shaped.flatten()

print('number of dots: {}'.format(np.shape(CoefClass.ShapeRemember)[0]))

'''
Construct right hand side
'''

f_ref = np.ones(NpFine) * 0.0001
f_ref_reshaped = f_ref.reshape(NFine+1)
# f_ref_reshaped[int(0*fine/8):int(2*fine/8),int(0*fine/8):int(2*fine/8)] = 1
# f_ref_reshaped[int(6*fine/8):int(8*fine/8),int(6*fine/8):int(8*fine/8)] = 1
f_ref_reshaped[int(1*fine/8):int(7*fine/8),int(1*fine/8):int(7*fine/8)] = 1
f_ref = f_ref_reshaped.reshape(NpFine)


'''
Perturbation using buildcoef2d
'''

# decision
valc = np.shape(CoefClass.ShapeRemember)[0]
numbers = []
decision = np.zeros(50)
decision[0] = 1


for i in range(0,valc):
    a = random.sample(list(decision),1)[0]
    if a == 1:
        numbers.append(i)

# ATTENTION : In this case TRANS means PERTURBED
aFine_trans = CoefClass.SpecificVanish(Number = numbers).flatten()

'''
Plot diffusion coefficient and right hand side
'''

plt.figure("Coefficient")
drawCoefficient_origin(NFine, aFine_ref)

plt.figure("Perturbed coefficient")
drawCoefficient_origin(NFine, aFine_trans)

plt.figure('Right hand side')
drawCoefficient_origin(NFine+1, f_ref)

plt.show()

'''
Compute FEM
'''

uFineFull_trans, AFine_trans, _ = femsolver.solveFine(world, aFine_trans, f_ref, None, boundaryConditions)


'''
Set the coefficient that we want to approximate and the tolerance
'''

a_Fine_to_be_approximated = aFine_trans

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


Algorithm = algorithms.PercentageVsErrorAlgorithm(world = world,
                                                 k = k ,
                                                 boundaryConditions = boundaryConditions,
                                                 a_Fine_to_be_approximated = a_Fine_to_be_approximated,
                                                 aFine_ref = aFine_ref,
                                                 f_trans = f_ref,
                                                 epsCoarse = epsCoarse,
                                                 KmsijT = KmsijT,
                                                 correctorsListT = correctorsListT,
                                                 patchT = patchT,
                                                 RFull = RFull,
                                                 Rf = Rf,
                                                 MFull = MFull,
                                                 uFineFull_trans = uFineFull_trans,
                                                 AFine_trans = AFine_trans)

to_be_updatedT, energy_errorT, tmp_errorT, TOLt, uFineFull_trans_LOD = Algorithm.StartAlgorithm()

store_all_data(ROOT, k, N, epsCoarse, to_be_updatedT, energy_errorT, tmp_errorT, TOLt, uFineFull_trans, uFineFull_trans_LOD, NFine, NWorldCoarse, aFine_ref, aFine_trans, f_ref, name="test2_perc")

'''
Plot solutions
'''
fig = plt.figure('Solutions')
ax = fig.add_subplot(121)
ax.set_title('Fem Solution to transformed problem',fontsize=6)
ax.imshow(np.reshape(uFineFull_trans, NFine+1), origin='lower_left')

ax = fig.add_subplot(122)
ax.set_title('PGLOD Solution to transformed problem',fontsize=6)
ax.imshow(np.reshape(uFineFull_trans_LOD, NFine+1), origin='lower_left')

'''
Errors
'''
energy_norm = np.sqrt(np.dot(uFineFull_trans_LOD, AFine_trans * uFineFull_trans_LOD))
energy_error = np.sqrt(np.dot((uFineFull_trans - uFineFull_trans_LOD), AFine_trans * (uFineFull_trans - uFineFull_trans_LOD)))
print("Energy norm {}, error {}, rel. error {}".format(energy_norm, energy_error, energy_error/energy_norm))

plt.show()