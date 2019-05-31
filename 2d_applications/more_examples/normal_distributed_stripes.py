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
from gridlod_on_perturbations.visualization_tools import d3sol
from MasterthesisLOD.visualize import drawCoefficientGrid, drawCoefficient


fine = 256
N = 16
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)

space = 20
thick = 2

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


# Set reference coefficient
aFine_ref_shaped = CoefClass.BuildCoefficient()
aFine_ref_shaped = CoefClass.SpecificMove(Number=np.arange(0,10), steps=4, Right=1)
aFine_ref = aFine_ref_shaped.flatten()
number_of_channels = len(CoefClass.ShapeRemember)

# Discrete mapping
Nmapping = np.array([int(fine),int(fine)])
cq1 = np.zeros((int(fine)+1,int(fine)+1))

size_of_an_element = 1./fine
walk_with_perturbation = size_of_an_element

channels_position_from_zero = space
channels_end_from_zero = channels_position_from_zero + thick

#I want to know the exact places of the channels
ref_array = aFine_ref_shaped[0]

eps_range = 0.03
for c in range(number_of_channels):
    now = 0
    count = 0
    for i in range(np.size(ref_array)):
        if ref_array[i] == 1:
            count +=1
        if count == (c+1)*thick:
            begin = i + 1 - space // 2
            end = i + 1 + thick+ space // 2
            break

    left = begin
    right = end
    increasing_length = 3
    constant_length = (end-begin) - increasing_length * 2
    epsilon = np.random.uniform(-eps_range, eps_range)
    walk = epsilon
    for i in range(increasing_length):
        cq1[:, begin+1+i] = (i+1)/increasing_length * walk
        cq1[:, begin + increasing_length + i + constant_length] = walk - (i+1)/increasing_length * walk

    for i in range(constant_length):
        cq1[:, begin + increasing_length + i] = walk

plt.plot(np.arange(0,fine+1),cq1[0,:], label= '$id(x) - \psi(x)$')
plt.plot(np.arange(0,fine),ref_array*0.01)
plt.title('Domain mapping')
plt.legend()
cq1 = cq1.flatten()

xpFine = util.pCoordinates(NFine)
xtFine = util.tCoordinates(NFine)

alpha = 1.

for_mapping = np.stack((xpFine[:,0] + alpha * func.evaluateCQ1(Nmapping, cq1, xpFine), xpFine[:,1]), axis = 1)
psi = discrete_mapping.MappingCQ1(NFine, for_mapping)


# Compute grid points and mapped grid points
# Grid naming:
# ._pert   is the grid mapped from reference to perturbed domain
# ._ref    is the grid mapped from perturbed to reference domain
xpFine_pert = psi.evaluate(xpFine)
xpFine_ref = psi.inverse_evaluate(xpFine)

xtFine_pert = psi.evaluate(xtFine)
xtFine_ref = psi.inverse_evaluate(xtFine)


# Compute perturbed coefficient
# Coefficient and right hand side naming:
# ._pert    is a function defined on the uniform grid in the perturbed domain
# ._ref     is a function defined on the uniform grid in the reference domain
# ._trans   is a function defined on the uniform grid on the reference domain,
#           after transformation from the perturbed domain
aFine_pert = func.evaluateDQ0(NFine, aFine_ref, xtFine_ref)
aBack_ref = func.evaluateDQ0(NFine, aFine_pert, xtFine_pert)

is_this_invertible = np.linalg.norm(aBack_ref-aFine_ref)
if is_this_invertible > 0.001:
    print('Psi is not invertible')
else:
    print('Psi is invertible')

plt.figure("Coefficient")
drawCoefficient(NFine, aFine_ref)

plt.figure("a_perturbed")
drawCoefficient(NFine, aFine_pert)

plt.figure("a_back")
drawCoefficient(NFine, aBack_ref)

plt.show()