# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, femsolver, interp, coef, fem, lod, pglod
from gridlod.world import World, Patch

from MasterthesisLOD import buildcoef2d
from gridlod_on_perturbations.visualization_tools import drawCoefficient_origin
from MasterthesisLOD.visualize import drawCoefficientGrid

import perturbations
import algorithms
from gridlod_on_perturbations.data import safe_data

ROOT = '../../2d_applications/data/test_from_scratch/'

# Set global variables for the computation

power = 8
fine = 2**power
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)

# This factor enables that A does not change for finer fine mesh
factor = 2**(power - 8)

# Coarse mesh and localization Parameter
N = 2**4
k = 3
NWorldCoarse = np.array([N, N])

# boundary Conditions
boundaryConditions = np.array([[0, 0], [0, 1]])

NCoarseElement = NFine // NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

'''
Construct right hand side
'''

f_ref = np.ones(NpFine) * 0.01
f_ref_reshaped = f_ref.reshape(NFine+1)
# f_ref_reshaped[int(0*fine/8):int(2*fine/8),int(0*fine/8):int(2*fine/8)] = 1
# f_ref_reshaped[int(6*fine/8):int(8*fine/8),int(6*fine/8):int(8*fine/8)] = 1
f_ref_reshaped[int(3*fine/8):int(5*fine/8),int(3*fine/8):int(5*fine/8)] = 10
f_ref = f_ref_reshaped.reshape(NpFine)

'''
Construct diffusion coefficient
'''

space = int(6 * factor)
thick = int(3 * factor)
bg = 0.1		#background
val = 1			#values

# See my master thesis Chapter 7 for detailed explanation on buildcoef2d
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
                        ChannelVertical     = None,  # SET THIS TO TRUE TO GET CHANNELS
                        BoundarySpace       = True)

# Set reference coefficient
aFine_ref = CoefClass.BuildCoefficient().flatten()


'''
Domain mapping perturbation
'''

bending_perturbation = perturbations.BendingInTwoAreas(world)
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
drawCoefficient_origin(NFine+1, f_ref)

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
