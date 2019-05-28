# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import random
import matplotlib.pyplot as plt

from gridlod import util, femsolver, func, interp, coef, fem, lod, pglod
from gridlod.world import World, Patch

from MasterthesisLOD import buildcoef2d
from gridlod_on_perturbations import discrete_mapping
from gridlod_on_perturbations.visualization_tools import d3sol, drawCoefficient_origin
from MasterthesisLOD.visualize import drawCoefficientGrid, drawCoefficient
import csv

# Set global variables for the computation

potenz = 8
factor = 2**(potenz - 8)
fine = 2**potenz
N = 2**2
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)

NWorldCoarse = np.array([N, N])
boundaryConditions = np.array([[0, 0], [0, 0]])
NCoarseElement = NFine // NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

# Construct diffusion coefficient
space = 20 * factor
thick = 10 * factor
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
aFine_ref = CoefClass.BuildCoefficient().flatten()

'''
Plot diffusion coefficient
'''
plt.figure("Coefficient")
drawCoefficient_origin(NFine, aFine_ref)


# right hand side
f_ref = np.ones(NpFine) * 0.001
f_ref_reshaped = f_ref.reshape(NFine+1)
f_ref_reshaped[:int(fine/4),:int(fine/4)] = 1
f_ref = f_ref_reshaped.reshape(NpFine)

'''
Plot right hand side
'''
plt.figure('Right hand side')
drawCoefficient_origin(NFine+1, f_ref)

a_Fine_to_be_approximated = aFine_ref

'''
Compute FEM
'''
uFineFull_ref, AFine_ref, _ = femsolver.solveFine(world, a_Fine_to_be_approximated, f_ref, None, boundaryConditions)



'''
Compute PGLOD 
'''
k = 2

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

# Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
print('compute KmsijT and RmsiT for all T')
patchT, correctorsListT, KmsijT, csiT = zip(*map(computeKmsij, range(world.NtCoarse)))
patchT, correctorRhsT, RmsiT = zip(*map(computeRmsi, range(world.NtCoarse)))

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