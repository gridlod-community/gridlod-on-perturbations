# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, femsolver, func, interp, coef, fem, lod, pglod
from gridlod.world import World, Patch

from MasterthesisLOD import buildcoef2d
from gridlod_on_perturbations import discrete_mapping
from gridlod_on_perturbations.visualization_tools import d3sol
from MasterthesisLOD.visualize import drawCoefficientGrid, drawCoefficient


fine = 128
N = 16
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)

space = 10
thick = 1

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
aFine_ref = aFine_ref_shaped.flatten()
number_of_channels = len(CoefClass.ShapeRemember)
number_of_perturbed_channels = 6
#I want to know the exact places of the channels
ref_array = aFine_ref_shaped[0]
now = 0
count = 0
for i in range(np.size(ref_array)):
    if ref_array[i] == 1:
        count +=1
    if count == number_of_channels//2-number_of_perturbed_channels//2+1:
        begin = i+1
        break
count = 0
for i in range(np.size(ref_array)):
    if ref_array[i] == 1:
        count +=1
    if count == number_of_channels//2 + number_of_perturbed_channels//2+1:
        end = i
        break

# Discrete mapping
Nmapping = np.array([int(fine),int(fine)])
cq1 = np.zeros((int(fine)+1,int(fine)+1))

size_of_an_element = 1./fine
walk_with_perturbation = size_of_an_element

channels_position_from_zero = space
channels_end_from_zero = channels_position_from_zero + thick

left = begin
right = end
increasing_length = (end-begin)//number_of_perturbed_channels - thick - 1
constant_length = (end-begin) - increasing_length * 2
maximum_walk = (increasing_length-1) * walk_with_perturbation
walk_with_perturbation = maximum_walk
for i in range(increasing_length):
    cq1[:, begin+1+i] = (i+1)/increasing_length * walk_with_perturbation
    cq1[:, begin + increasing_length + i + constant_length] = walk_with_perturbation - (i+1)/increasing_length * walk_with_perturbation

for i in range(constant_length):
    cq1[:, begin + increasing_length + i] = walk_with_perturbation

plt.plot(np.arange(0,fine+1),cq1[0,:], label= '$id(x) - \psi(x)$')
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

plt.figure("Coefficient")
drawCoefficient(NFine, aFine_ref)

plt.figure("a_perturbed")
drawCoefficient(NFine, aFine_pert)

plt.figure("a_back")
drawCoefficient(NFine, aBack_ref)


plt.show()