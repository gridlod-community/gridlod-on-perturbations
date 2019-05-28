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

# try around with f
# CoefClass_for_rhs = buildcoef2d.Coefficient2d(NFine,
#                         bg                  = 0.01,
#                         val                 = val,
#                         length              = 8,
#                         thick               = 8,
#                         space               = 8,
#                         probfactor          = 1,
#                         right               = 1,
#                         down                = 0,
#                         diagr1              = 0,
#                         diagr2              = 0,
#                         diagl1              = 0,
#                         diagl2              = 0,
#                         LenSwitch           = None,
#                         thickSwitch         = None,
#                         equidistant         = True,
#                         ChannelHorizontal   = None,
#                         ChannelVertical     = None,
#                         BoundarySpace       = True)
#
#
# f_cheat = CoefClass_for_rhs.BuildCoefficient().flatten() *10.
# f_cheat = f_cheat.reshape(fine, fine)
# f_cheat = np.append(f_cheat, np.ones((fine, 1)) * bg, 1)
# f_row = np.ones((1, fine + 1))
# f_row[0] = f_cheat[0, :]
# f_cheat = np.append(f_cheat, f_row, 0)
# f_cheat = f_cheat.flatten()

# plt.figure('check fine f')
# drawCoefficient_origin(NFine + 1, f_cheat)
# plt.show()


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

