# This file is part of the OWR report "Multiscale methods for perturbed diffusion problems":
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

import gridlod_on_perturbations.algorithms as algorithms
from gridlod_on_perturbations.data import store_all_data

from gridlod_on_perturbations.visualization_tools import draw_f, draw_indicator

ROOT = '../../2d_applications/data/HeKeMa_2019/exOWR'

# Set global variables for the computation

potenz = 8
factor = 2**(potenz - 8)
fine = 2**potenz

store = True
name = 'test'
N = 2**5
print('log H: ' ,np.abs(np.log(np.sqrt(2*(1./N**2)))))
k = 4  # goes like log H

NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)
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

bg = 0.1		#background
val = 1			#values

CoefClass = buildcoef2d.Coefficient2d(NFine,
                        bg                  = bg,
                        val                 = val,
                        length              = thick,
                        thick               = thick,
                        space               = space,
                        probfactor          = 1,
                        right               = 1,
                        equidistant         = True,
                        BoundarySpace       = True)


# Set reference coefficient
aFine_ref_shaped = CoefClass.BuildCoefficient()
# I want to have the dots in the middle of the domain.
# aFine_ref = CoefClass.SpecificMove(Number=np.arange(CoefClass.shapecounter), Right=1 , steps=4).flatten()
aFine_ref = aFine_ref_shaped.flatten()

print('number of dots: {}'.format(CoefClass.shapecounter))

'''
Construct right hand side
'''

f_ref = np.zeros(NpFine)
f_ref_reshaped = f_ref.reshape(NFine+1)
f_ref_reshaped[int(1*fine/8)+1:int(7*fine/8),int(1*fine/8)+1:int(7*fine/8)] = 1
f_ref = f_ref_reshaped.reshape(NpFine)

f_trans = f_ref

'''
compute norm of f
'''

MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
norm_of_f = [np.sqrt(np.dot(f_trans, MFull * f_trans))]
print('Norm of f is {}'.format(norm_of_f[0]))


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
    a = random.sample(list(decision), 1)[0]
    if a == 1:
        numbers.append(i)

# ATTENTION : In this case TRANS means PERTURBED
aFine_trans = CoefClass.SpecificVanish(Number = numbers, Original=False).flatten()

'''
Plot diffusion coefficient and right hand side
'''

plt.figure("Coefficient")
drawCoefficient_origin(NFine, aFine_ref)

plt.figure("Perturbed coefficient")
drawCoefficient_origin(NFine, aFine_trans)

plt.figure('Right hand side')
draw_f(NFine+1, f_ref)

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

def real_computeKmsij(TInd):
    print('.', end='', flush=True)
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    aPatch = lambda: coef.localizeCoefficient(patch, a_Fine_to_be_approximated)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

def real_computeRmsi(TInd):
    print('.', end='', flush=True)
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    aPatch = lambda: coef.localizeCoefficient(patch, a_Fine_to_be_approximated)
    MRhsList = [f_ref[util.extractElementFine(world.NWorldCoarse,
                                          world.NCoarseElement,
                                          patch.iElementWorldCoarse,
                                          extractElements=False)]];

    correctorRhs = lod.computeElementCorrector(patch, IPatch, aPatch, None, MRhsList)[0]
    Rmsi, cetaTPrime = lod.computeRhsCoarseQuantities(patch, correctorRhs, aPatch, True)

    return patch, correctorRhs, Rmsi, cetaTPrime


def computeKmsij(TInd):
    print('.', end='', flush=True)
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    aPatch = lambda: coef.localizeCoefficient(patch, aFine_ref)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

def computeRmsi(TInd):
    print('.', end='', flush=True)
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    aPatch = lambda: coef.localizeCoefficient(patch, aFine_ref)
    MRhsList = [f_ref[util.extractElementFine(world.NWorldCoarse,
                                          world.NCoarseElement,
                                          patch.iElementWorldCoarse,
                                          extractElements=False)]];

    correctorRhs = lod.computeElementCorrector(patch, IPatch, aPatch, None, MRhsList)[0]
    Rmsi, cetaTPrime = lod.computeRhsCoarseQuantities(patch, correctorRhs, aPatch, True)

    return patch, correctorRhs, Rmsi, cetaTPrime

def computeIndicators(TInd):
    print('.', end='', flush=True)
    aPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine_ref)
    rPatch = lambda: coef.localizeCoefficient(patchT[TInd], a_Fine_to_be_approximated)

    E_vh = lod.computeErrorIndicatorCoarseFromCoefficients(patchT[TInd], csiT[TInd].muTPrime,  aPatch, rPatch)
    E_vh *= norm_of_f[0]

    # this is new for E_ft
    f_ref_patch = f_ref[util.extractElementFine(world.NWorldCoarse,
                                          world.NCoarseElement,
                                          patchT[TInd].iElementWorldCoarse,
                                          extractElements=False)]
    f_patch = f_trans[util.extractElementFine(world.NWorldCoarse,
                                          world.NCoarseElement,
                                          patchT[TInd].iElementWorldCoarse,
                                          extractElements=False)]
    E_f, E_Rf = lod.computeEftErrorIndicatorsCoarse(patchT[TInd], cetaTPrimeT[TInd], aPatch, rPatch, f_ref_patch, f_patch)

    return E_vh, E_f, E_Rf



print('precomputing ....')

# Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
print('computing real correctors',  end='', flush=True)
patchT, correctorsListT, KmsijT, csiT = zip(*map(real_computeKmsij, range(world.NtCoarse)))
print()
print('computing real right hand side correctors',  end='', flush=True)
patchT, correctorRhsT, RmsiT, cetaTPrimeT = zip(*map(real_computeRmsi, range(world.NtCoarse)))
print()

KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
RFull = pglod.assemblePatchFunction(world, patchT, RmsiT)
MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)

Rf = pglod.assemblePatchFunction(world, patchT, correctorRhsT)

basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)

bFull = basis.T * MFull * f_trans # - RFull

basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
modifiedBasis = basis - basisCorrectors

uFull, _ = pglod.solve(world, KFull, bFull, boundaryConditions)

uLodFine = modifiedBasis * uFull
# uLodFine += Rf

u_best_LOD = uLodFine

energy_norm = np.sqrt(np.dot(u_best_LOD, AFine_trans * u_best_LOD))
energy_error = np.sqrt(np.dot((uFineFull_trans - u_best_LOD), AFine_trans * (uFineFull_trans - u_best_LOD)))
print("Energy norm {}, error {}, rel. error {}".format(energy_norm, energy_error, energy_error/energy_norm))


# Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
print('computing real correctors',  end='', flush=True)
patchT, correctorsListT, KmsijT, csiT = zip(*map(computeKmsij, range(world.NtCoarse)))
print()
print('computing real right hand side correctors',  end='', flush=True)
patchT, correctorRhsT, RmsiT, cetaTPrimeT = zip(*map(computeRmsi, range(world.NtCoarse)))
print()


RFull = pglod.assemblePatchFunction(world, patchT, RmsiT)
MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)

print('computing error indicators',  end='', flush=True)
E_vh, E_fT, E_RfT = zip(*map(computeIndicators, range(world.NtCoarse)))
print()


'''
Plot error indicators
'''
np_eps = np.einsum('i,i -> i', np.ones(np.size(E_vh)), E_vh)
draw_indicator(NWorldCoarse, np_eps, original_style=True, Gridsize=N)

np_eft = np.einsum('i,i -> i', np.ones(np.size(E_fT)), E_fT)
draw_indicator(NWorldCoarse, np_eft, original_style=True, Gridsize=N, string='eft')

np_eRft = np.einsum('i,i -> i', np.ones(np.size(E_RfT)), E_RfT)
draw_indicator(NWorldCoarse, np_eRft, original_style=True, Gridsize=N, string='eRft')


plt.show()


Algorithm = algorithms.PercentageVsErrorAlgorithm(world = world,
                                                 k = k ,
                                                 boundaryConditions = boundaryConditions,
                                                 a_Fine_to_be_approximated = a_Fine_to_be_approximated,
                                                 aFine_ref = aFine_ref,
                                                 f_trans = f_trans,
                                                 E_vh = E_vh,
                                                 KmsijT = KmsijT,
                                                 correctorsListT = correctorsListT,
                                                 patchT = patchT,
                                                 RmsijT=RmsiT,
                                                 correctorsRhsT = correctorRhsT,
                                                 MFull = MFull,
                                                 uFineFull_trans = uFineFull_trans,
                                                 AFine_trans = AFine_trans,
                                                 compare_with_best_LOD = True,
                                                 u_best_LOD = u_best_LOD)

to_be_updatedT, energy_errorT, tmp_errorT, rel_errorT, TOLt, uFineFull_trans_LOD = Algorithm.StartAlgorithm()

if store:
    store_all_data(ROOT, k, N, E_vh, np_eft, np_eRft, norm_of_f, to_be_updatedT, energy_errorT, tmp_errorT, rel_errorT, TOLt, uFineFull_trans, uFineFull_trans_LOD, NFine, NWorldCoarse, aFine_ref, aFine_trans,  f_ref, aFine_trans, f_trans, name=name)

energy_error = np.sqrt(np.dot((uFineFull_trans_LOD - u_best_LOD), AFine_trans * (uFineFull_trans_LOD - u_best_LOD)))
print("Energy norm {}, error {}, rel. error {}".format(energy_norm, energy_error, energy_error/energy_norm))


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
