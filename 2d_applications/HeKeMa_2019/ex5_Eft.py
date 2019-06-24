# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import random
import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, femsolver, interp, coef, fem, lod, pglod
from gridlod.world import World, Patch

from MasterthesisLOD import buildcoef2d

from visualization_tools import draw_f, draw_indicator, drawCoefficient_origin

import perturbations
import algorithms
from gridlod_on_perturbations.data import store_all_data

ROOT = '../../2d_applications/data/HeKeMa_2019/ex5'

# Set global variables for the computation

potenz = 8
factor = 2**(potenz - 8)
fine = 2**potenz

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

space = 6 * factor
thick = 0 * factor    # This is why the upper coefficient is only background.

bg = 0.1		#background
val = 1			#values

soilinput = np.array([[13, 6, 3]])   # this means 13 rows with space 6 fine elements and dots of size 3 fine elements
# soilinput = np.array([[13, 3, 6]])   # this means 13 rows with space 3 fine elements and dots of size 6 fine elements
soilMatrix = buildcoef2d.soil_converter(soilinput, NFine, BoundarySpace=space)
print(soilMatrix)   #

CoefClass = buildcoef2d.Coefficient2d(NFine,
                        bg                  = bg,
                        val                 = val,
                        length              = thick,
                        thick               = thick,
                        space               = space,
                        probfactor          = 1,
                        right               = 1,
                        equidistant         = True,
                        BoundarySpace       = True,
                        soilMatrix          = soilMatrix)


# Set reference coefficient
aFine_ref_shaped = CoefClass.BuildCoefficient()
aFine_ref = aFine_ref_shaped.flatten()

'''
Construct right hand side
'''

f_ref = np.ones(NpFine) * 0.0001
f_ref_reshaped = f_ref.reshape(NFine+1)
f_ref_reshaped[int(8*fine/40):int(11*fine/40),int(19*fine/40):int(22*fine/40)] = 10
f_ref_reshaped[int(30*fine/40):int(33*fine/40),int(19*fine/40):int(22*fine/40)] = 10
f_ref = f_ref_reshaped.reshape(NpFine)



'''
Domain mapping perturbation
'''
# in order to change psi you can just change those two variables
area = [0,1]
bending_factor = 0.25

bending_perturbation = perturbations.BendingInOneArea(world, area=area, bending_factor=bending_factor)
aFine_pert, f_pert = bending_perturbation.computePerturbation(aFine_ref, f_ref)
aFine_trans, f_trans = bending_perturbation.computeTransformation(aFine_ref, f_ref)


'''
Plot diffusion coefficient and right hand side
'''

plt.figure("Coefficient")
drawCoefficient_origin(NFine, aFine_ref)

plt.figure("Perturbed coefficient")
drawCoefficient_origin(NFine, aFine_pert)

plt.figure('transformed')
drawCoefficient_origin(NFine, aFine_trans)

plt.figure('Right hand side')
draw_f(NFine+1, f_ref)

plt.show()

'''
Check whether domain mapping method works sufficiently good
'''

uFineFull_pert, AFine_pert, _ = femsolver.solveFine(world, aFine_pert, f_pert, None, boundaryConditions)
uFineFull_trans, AFine_trans, _ = femsolver.solveFine(world, aFine_trans, f_trans, None, boundaryConditions)

u_FineFull_trans_pert = bending_perturbation.evaluateSolution(uFineFull_trans)

energy_norm = np.sqrt(np.dot(uFineFull_pert, AFine_pert * uFineFull_pert))
energy_error = np.sqrt(np.dot((u_FineFull_trans_pert - uFineFull_pert),
                              AFine_pert * (u_FineFull_trans_pert - uFineFull_pert)))
print("Domain Mapping with FEM: Energy norm {}, error {}, rel. error {}".format(energy_norm, energy_error,
                                                       energy_error / energy_norm))

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
    aPatch = lambda: coef.localizeCoefficient(patch, aFine_ref)
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

    # E_fT here !
    E_fT = np.zeros(np.shape(epsCoarse)) # so far only zeros

    return epsCoarse, E_fT



print('precomputing ....')

# Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
patchT, correctorsListT, KmsijT, csiT = zip(*map(computeKmsij, range(world.NtCoarse)))
patchT, correctorRhsT, RmsiT = zip(*map(computeRmsi, range(world.NtCoarse)))


RFull = pglod.assemblePatchFunction(world, patchT, RmsiT)
MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)

print('computing error indicators')
epsCoarse, E_fT = zip(*map(computeIndicators, range(world.NtCoarse)))

'''
Plot error indicators
'''
np_eps = np.einsum('i,i -> i', np.ones(np.size(epsCoarse)), epsCoarse)
draw_indicator(NWorldCoarse, np_eps, original_style=True, Gridsize=N)

np_eft = np.einsum('i,i -> i', np.ones(np.size(E_fT)), E_fT)
draw_indicator(NWorldCoarse, np_eft, original_style=True, Gridsize=N, string='eft')

plt.show()

Algorithm = algorithms.AdaptiveAlgorithm(world = world,
                                                 k = k ,
                                                 boundaryConditions = boundaryConditions,
                                                 a_Fine_to_be_approximated = a_Fine_to_be_approximated,
                                                 aFine_ref = aFine_ref,
                                                 f_trans = f_trans,
                                                 epsCoarse = epsCoarse,
                                                 KmsijT = KmsijT,
                                                 correctorsListT = correctorsListT,
                                                 patchT = patchT,
                                                 RmsijT=RmsiT,
                                                 correctorsRhsT = correctorRhsT,
                                                 MFull = MFull,
                                                 uFineFull_trans = uFineFull_trans,
                                                 AFine_trans = AFine_trans #)
                                                 ,StartingTolerance=0)

to_be_updatedT, energy_errorT, tmp_errorT, rel_energy_errorT, TOLt, uFineFull_trans_LOD = Algorithm.StartAlgorithm()

store_all_data(ROOT, k, N, epsCoarse, to_be_updatedT, energy_errorT, tmp_errorT, rel_energy_errorT, TOLt, uFineFull_trans, uFineFull_trans_LOD, NFine, NWorldCoarse, aFine_ref, aFine_pert,  f_ref, aFine_trans, f_trans, np_eft)

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