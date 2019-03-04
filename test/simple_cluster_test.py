# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, femsolver, func, interp, coef, fem, pg
from gridlod.world import World

from MasterthesisLOD import pg_pert, buildcoef2d
from gridlod_on_perturbations.visualization_tools import drawCoefficient_origin, d3sol

fine = 256
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

# Setting up PGLOD
IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, boundaryConditions)
a_fine_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, a_fine)

pglod = pg_pert.PerturbedPetrovGalerkinLOD(a_fine_coef, world, 2, IPatchGenerator, 3, slurmCluster=True)

# compute correctors
pglod.originCorrectors(clearFineQuantities=False)

#solve upscaled system
uLodFine, _, _ = pglod.solve(f)

energy_norm = np.sqrt(np.dot(uLodFine - uFineFull, AFine * (uLodFine - uFineFull)))
print(energy_norm)

#plt.show()
