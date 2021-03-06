# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from gridlod import pglod, util, lod, interp, coef, fem, femsolver
from gridlod.world import World, Patch

from gridlod_on_perturbations.visualization_tools import drawCoefficient_origin, draw_f

from MasterthesisLOD import buildcoef2d

potenz = 8
factor = 2**(potenz - 8)
fine = 2**potenz
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)

space = 50#int(6 * factor)
thick = 50#int(6 * factor)

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


aFine = CoefClass.BuildCoefficient().flatten()

# for poisson !
# aFine = np.ones(np.shape(aFine))

NWorldCoarse = np.array([2,2])
NCoarseElement = NFine // NWorldCoarse
boundaryConditions = np.array([[0, 0], [0, 0]])
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)


f = np.zeros(NpFine) #* 0.0001
f_reshaped = f.reshape(NFine+1)
# f_ref_reshaped[int(0*fine/8):int(2*fine/8),int(0*fine/8):int(2*fine/8)] = 1
# f_ref_reshaped[int(6*fine/8):int(8*fine/8),int(6*fine/8):int(8*fine/8)] = 1
f_reshaped[int(4*fine/8):int(5*fine/8),int(4*fine/8):int(5*fine/8)] = 10
f = f_reshaped.reshape(NpFine)

plt.figure("Coefficient")
drawCoefficient_origin(NFine, aFine)

plt.figure('Right hand side')
draw_f(NFine+1, f)

plt.show()

uSol, AFine, _ = femsolver.solveFine(world, aFine, f, None, boundaryConditions)

kList = [4]
NList = [32]
logH = {N: np.abs(np.log(np.sqrt(2*(1./N**2)))) for N in NList}
for N in NList:
    print('___________________________________________________')
    for k in kList:
        print('   k = {}    H = 1/{}    logH = {:3f}   => k = {} should be sufficient'.format(
            k,N,logH[N],int(logH[N]+0.99)), end='', flush=True)
        NWorldCoarse = np.array([N,N])
        NCoarseElement = NFine // NWorldCoarse
        boundaryConditions = np.array([[0, 0], [0, 0]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        xpCoarse = util.pCoordinates(NWorldCoarse).flatten()


        def computeKmsij(TInd):
            print('.', end='', flush=True)
            patch = Patch(world, k, TInd)
            IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
            aPatch = lambda: coef.localizeCoefficient(patch, aFine)

            correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
            csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
            return patch, correctorsList, csi.Kmsij


        def computeRmsi(TInd):
            print('.', end='', flush=True)
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
        print('!')
        patchT, correctorRhsT, RmsiT = zip(*map(computeRmsi, range(world.NtCoarse)))
        print('!')
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

        norm = np.sqrt(np.dot(uLodFine, AFine * (uLodFine)))


        print('                                    Normal: {:8f}    Relative: {:8f}'.format(newErrorFine, newErrorFine/norm))


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

        norm = np.sqrt(np.dot(uLodFine, AFine * (uLodFine)))

        print('                       With RHS correction: {:8f}    Relative: {:8f}'.format(
            newErrorFine, newErrorFine / norm))

