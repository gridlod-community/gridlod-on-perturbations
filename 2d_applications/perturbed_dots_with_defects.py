# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import print_function

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

space = 8
thick = 4

bg = 0.01 		#background
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


# Set reference coefficient
aCoarse_ref_shaped = CoefClass.BuildCoefficient()

aCoarse_ref = aCoarse_ref_shaped.flatten()
aFine_ref = aCoarse_ref

to_be_perturbed = CoefClass.SpecificVanish(Number=[1,22,30,69, 90])
# Discrete mapping
Nmapping = np.array([int(fine),int(fine)])

size_of_an_element = 1./128.
steps =  3
maximum_walk_with_perturbation =  steps * size_of_an_element
number_of_dots_in_one_dimension = np.sqrt(len(CoefClass.ShapeRemember))

dots_position_from_zero = space

np.random.seed(4)

cq1 = np.zeros((int(fine)+1,int(fine)+1))

for i in range(0,10,2):
    position = dots_position_from_zero * (i+1) + i * thick + int(round(thick/2. - 0.9))
    left = position-space/2 + 1 - space - thick
    right = position+thick + space/2 - 1 + space + thick
    step = np.random.random_integers(0,steps,1)
    sign = (-1)**np.random.random_integers(0,1,1)
    print(step * sign)
    #now the y axis is also random
    j = int(np.random.random_integers(2,number_of_dots_in_one_dimension-2,1))
    position_y = dots_position_from_zero * (j + 1) + j * thick + int(round(thick / 2. - 0.9))
    left_y = position_y-space/2 + 1 - space - thick
    right_y = position_y+thick + space/2 - 1 + space + thick
    cq1[left_y:right_y, left:right] = step * sign * size_of_an_element

cq1 = cq1.flatten()

cq2 = np.zeros((int(fine)+1,int(fine)+1))

for i in range(0,10,2):
    position = dots_position_from_zero * (i+1) + i * thick + int(round(thick/2. - 0.9))
    left = position-space/2 + 1 - space - thick
    right = position+thick + space/2 - 1 + space + thick
    step = np.random.random_integers(0,steps,1)
    sign = (-1)**np.random.random_integers(0,1,1)
    print(step * sign)
    #now the y axis is also random
    j = int(np.random.random_integers(2,number_of_dots_in_one_dimension-2,1))
    position_y = dots_position_from_zero * (j + 1) + j * thick + int(round(thick / 2. - 0.9))
    left_y = position_y-space/2 + 1 - space - thick
    right_y = position_y+thick + space/2 - 1 + space + thick
    cq2[left_y:right_y, left:right] = step * sign * size_of_an_element

cq2 = cq2.flatten()

xpFine = util.pCoordinates(NFine)
xtFine = util.tCoordinates(NFine)

alpha = 1.

for_mapping = np.stack((xpFine[:,0] + alpha * func.evaluateCQ1(Nmapping, cq1, xpFine),
                        xpFine[:,1] + alpha * func.evaluateCQ1(Nmapping, cq2, xpFine)), axis = 1)
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
aFine_pert = func.evaluateDQ0(NFine, to_be_perturbed.flatten(), xtFine_ref)
aBack_ref = func.evaluateDQ0(NFine, aFine_pert, xtFine_pert)

plt.figure("Coefficient")
drawCoefficient_origin(NFine, aFine_ref)

plt.figure("a_perturbed")
drawCoefficient_origin(NFine, aFine_pert)

plt.figure("a_back")
drawCoefficient_origin(NFine, aBack_ref)

plt.show()