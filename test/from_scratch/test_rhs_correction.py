# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from gridlod import pglod, util, lod, interp, coef, fem, femsolver
from gridlod.world import World, Patch

from gridlod_on_perturbations.visualization_tools import drawCoefficient_origin

from MasterthesisLOD import buildcoef2d

factor = 2**0
fine = 128 * factor
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)

space = 16 * factor
thick = 2 * factor

bg = 0.1		#background
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


aFine = CoefClass.BuildCoefficient().flatten()

NWorldCoarse = np.array([2,2])
NCoarseElement = NFine // NWorldCoarse
boundaryConditions = np.array([[0, 0], [0, 0]])
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)


f = np.ones(world.NpFine) * 10.
#f = f_cheat
uSol, AFine, _ = femsolver.solveFine(world, aFine, f, None, boundaryConditions)

kList = [1,2,3,4]
NList = [2,4,8,16,32]
logH = {N: np.abs(np.log(np.sqrt(2*(1./N**2)))) for N in NList}
print(logH)
for N in NList:
    print('___________________________________________________')
    for k in kList:
        print('   k = {}    N = {}    logH = {}'.format(k,N,logH[N]))
        NWorldCoarse = np.array([N,N])
        NCoarseElement = NFine // NWorldCoarse
        boundaryConditions = np.array([[0, 0], [0, 0]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        xpCoarse = util.pCoordinates(NWorldCoarse).flatten()


        def computeKmsij(TInd):
            patch = Patch(world, k, TInd)
            IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
            aPatch = lambda: coef.localizeCoefficient(patch, aFine)

            correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
            csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
            return patch, correctorsList, csi.Kmsij


        def computeRmsi(TInd):
            patch = Patch(world, k, TInd)
            IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
            aPatch = lambda: coef.localizeCoefficient(patch, aFine)
            MRhsList = [f[util.extractElementFine(world.NWorldCoarse,
                                                     world.NCoarseElement,
                                                     patch.iElementWorldCoarse,
                                                     extractElements=False)]];

            correctorRhs = lod.computeElementCorrector(patch, IPatch, aPatch, None, MRhsList)[0]
            Rmsi = lod.computeRhsCoarseQuantities(patch, correctorRhs, aPatch)
            return patch, correctorRhs, Rmsi

        # Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
        patchT, correctorsListT, KmsijT = zip(*map(computeKmsij, range(world.NtCoarse)))
        patchT, correctorRhsT, RmsiT = zip(*map(computeRmsi, range(world.NtCoarse)))

        KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
        RFull = pglod.assemblePatchFunction(world, patchT, RmsiT)
        MFull = fem.assemblePatchMatrix(NFine, world.MLocFine)

        free = util.interiorpIndexMap(NWorldCoarse)
        basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)

        bFull = basis.T * MFull * f

        KFree = KFull[free][:, free]
        bFree = bFull[free]

        xFree = sparse.linalg.spsolve(KFree, bFree)

        basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
        basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
        modifiedBasis = basis - basisCorrectors
        xFull = np.zeros(world.NpCoarse)
        xFull[free] = xFree
        uLodCoarse = basis * xFull
        uLodFine = modifiedBasis * xFull

        AFine = fem.assemblePatchMatrix(NFine, world.ALocFine, aFine)
        MFine = fem.assemblePatchMatrix(NFine, world.MLocFine)

        newErrorCoarse = np.sqrt(np.dot(uSol - uLodCoarse, MFine * (uSol - uLodCoarse)))
        newErrorFine = np.sqrt(np.dot(uSol - uLodFine, AFine * (uSol - uLodFine)))

        print('                                    Normal: ', newErrorFine)


        bFull = basis.T * MFull * f - RFull

        KFree = KFull[free][:, free]
        bFree = bFull[free]

        xFree = sparse.linalg.spsolve(KFree, bFree)


        basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
        modifiedBasis = basis - basisCorrectors
        xFull = np.zeros(world.NpCoarse)
        xFull[free] = xFree
        uLodCoarse = basis * xFull
        uLodFine = modifiedBasis * xFull
        uLodFine += pglod.assemblePatchFunction(world, patchT, correctorRhsT)

        AFine = fem.assemblePatchMatrix(NFine, world.ALocFine, aFine)

        newErrorFine = np.sqrt(np.dot(uSol - uLodFine, AFine * (uSol - uLodFine)))

        print('                       With RHS correction: ', newErrorFine)

