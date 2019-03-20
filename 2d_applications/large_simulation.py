# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from gridlod import util, femsolver, func, interp, coef, fem, lod, pglod
from gridlod.world import World, Patch

from MasterthesisLOD import buildcoef2d
from MasterthesisLOD.visualize import drawCoefficientGrid
import timeit

import ipyparallel as ipp

#client = ipp.Client(profile='slurm')
client = ipp.Client(sshserver='local')
client[:].use_cloudpickle()
view = client.load_balanced_view()

fine = 2**10
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)

N = 2**6

xpFine = util.pCoordinates(NFine)
xtFine = util.tCoordinates(NFine)


bg = 0.01 		#background
val = 1			#values

CoefClass = buildcoef2d.Coefficient2d(NFine,
                        bg                  = bg,
                        val                 = val,
                        length              = 2,
                        thick               = 2,
                        space               = 2,
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
                        BoundarySpace       = False)

aFine_shaped = CoefClass.BuildCoefficient()
aFine_ref = aFine_shaped.flatten()

f_ref = np.ones(np.prod(NFine+1))

NWorldCoarse = np.array([N, N])
boundaryConditions = np.array([[0, 0],[0, 0]])

NCoarseElement = NFine // NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

start_runtime = timeit.default_timer()

uFineFull_ref, AFine_ref, _ = femsolver.solveFine(world, aFine_ref, f_ref, None, boundaryConditions)

end_runtime = timeit.default_timer()
print('computing took {} seconds'.format(end_runtime-start_runtime))

k = 2
start_runtime = timeit.default_timer()

def computeKmsij(TInd):
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    aPatch = lambda: coef.localizeCoefficient(patch, aFine_ref)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

mid_element = N**2 //2 + N//2
mid_patch = Patch(world, k, mid_element)
IPatch = lambda: interp.L2ProjectionPatchMatrix(mid_patch, boundaryConditions)
aPatch = lambda: coef.localizeCoefficient(mid_patch, aFine_ref)
complete_size = np.size(aPatch())/np.prod(NCoarseElement)
periodic_correctorList = lod.computeBasisCorrectors(mid_patch, IPatch, aPatch)
periodic_csi = lod.computeBasisCoarseQuantities(mid_patch, periodic_correctorList, aPatch)

def computeKmsij_periodic(TInd):
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    aPatch = lambda: coef.localizeCoefficient(patch, aFine_ref)
    if np.size(aPatch())/np.prod(NCoarseElement) != complete_size:
        correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    else:
        correctorsList = periodic_correctorList
        csi = periodic_csi
    return patch, correctorsList, csi.Kmsij, csi

#periodic
patchT, correctorsListT, KmsijT, csiT = zip(*view.map_sync(computeKmsij_periodic, range(world.NtCoarse)))
#nonperiodic
#patchT, correctorsListT, KmsijT, csiT = zip(*map(computeKmsij, range(world.NtCoarse)))

end_runtime = timeit.default_timer()
print('computing took {} seconds'.format(end_runtime-start_runtime))

print('solve the system')
KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)

MFull = fem.assemblePatchMatrix(NFine, world.MLocFine)

basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
modifiedBasis = basis - basisCorrectors

bFull = MFull * f_ref
bFull = basis.T * bFull

uFull, _ = pglod.solve(world, KFull, bFull, boundaryConditions)

uLodFine = modifiedBasis * uFull

newErrorFine = np.sqrt(np.dot(uLodFine - uFineFull_ref, AFine_ref * (uLodFine - uFineFull_ref)))

print('Error: {}'.format(newErrorFine))

print('finished')
