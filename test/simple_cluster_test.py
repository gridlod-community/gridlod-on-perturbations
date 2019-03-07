# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, femsolver, func, interp, coef, fem, pglod, lod
from gridlod.world import World, Patch

from MasterthesisLOD import buildcoef2d
from gridlod_on_perturbations.visualization_tools import drawCoefficient_origin, d3sol

import ipyparallel as ipp

client = ipp.Client(profile='slurm')
#client = ipp.Client(sshserver='local')
client[:].use_cloudpickle()
view = client.load_balanced_view()

fine = 32
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)

N = 16
bg = 0.01 		#background
val = 1			#values

CoefClass = buildcoef2d.Coefficient2d(NFine,
                        bg                  = bg,
                        val                 = val,
                        length              = 4,
                        thick               = 4,
                        space               = 4,
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

a_fine = CoefClass.BuildCoefficient().flatten()

plt.figure("Coefficient")
drawCoefficient_origin(NFine, a_fine)

f = np.ones(np.prod(NFine+1))

NWorldCoarse = np.array([N, N])
boundaryConditions = np.array([[0, 0],[0, 0]])

NCoarseElement = NFine // NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

uFineFull, AFine, _ = femsolver.solveFine(world, a_fine, f, None, boundaryConditions)

k = 2

def computeKmsij(TInd):
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    aPatch = lambda: coef.localizeCoefficient(patch, a_fine)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij


# Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
patchT, correctorsListT, KmsijT = zip(*view.map_sync(computeKmsij, range(world.NtCoarse)))

KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)

print('finished')