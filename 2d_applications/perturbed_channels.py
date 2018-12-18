# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)



import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse

from gridlod import util, world, fem, femsolver, func, interp, pg, coef
from gridlod.world import World

import psi_functions
import discrete_mapping
from visualization_tools import drawCoefficient, drawCoefficient_origin, d3plotter, d3solextra, d3sol
import buildcoef2d


fine = 128
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)

space = 16
thick = 4

bg = 0.01 		#background
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
aCoarse_ref_shaped = CoefClass.BuildCoefficient()

aCoarse_ref = aCoarse_ref_shaped.flatten()
aFine_ref = aCoarse_ref


# Discrete mapping
Nmapping = np.array([int(fine),int(fine)])
cq1 = np.zeros((int(fine)+1,int(fine)+1))

size_of_an_element = 1./128.
print(round(space//2 - thick//2 -0.9))
walk_with_perturbation =  round(space/2 - thick/2 - 0.9) * size_of_an_element
number_of_channels = len(CoefClass.ShapeRemember)

channels_position_from_zero = space
channels_end_from_zero = channels_position_from_zero + thick

for i in range(number_of_channels):
#for i in [0]:
    position = channels_position_from_zero * (i+1) + i * thick
    left = position-space//2 + 3
    right = position+thick + space//2-3
    print(left, position, right)
    if i%2 == 0:
        cq1[:, left:right] = walk_with_perturbation
    else:
        cq1[:, left:right] = - walk_with_perturbation


plt.plot(np.arange(0,129),cq1[0,:])
cq1 = cq1.flatten()

#print(cq1[:,10:15])
xpFine = util.pCoordinates(NFine)
xtFine = util.tCoordinates(NFine)


plt.figure('test')
plt.plot(xpFine[:,0], func.evaluateCQ1(Nmapping, cq1, xpFine))
#print(cq1)
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
drawCoefficient_origin(NFine, aFine_ref)

plt.figure("a_perturbed")
drawCoefficient_origin(NFine, aFine_pert)

plt.figure("a_back")
drawCoefficient_origin(NFine, aBack_ref)

plt.show()