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
from gridlod_on_perturbations.visualization_tools import d3sol, drawCoefficient_origin
from MasterthesisLOD.visualize import drawCoefficientGrid, drawCoefficient

potenz = 8
factor = 2**(potenz - 8)
fine = 2**potenz
N = 2**5
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)

space = 6 * factor
thick = 3 * factor

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


# Set reference coefficient
aFine_ref_shaped = CoefClass.BuildCoefficient()

# delete the second stripe that gets too much seperated
# aFine_ref_shaped = CoefClass.SpecificVanish(Number=[1])

aFine_ref = aFine_ref_shaped.flatten()

xpFine = util.pCoordinates(NFine)
xtFine = util.tCoordinates(NFine)

size_of_an_element = 1./fine
print('the size of a fine element is {}'.format(size_of_an_element))
walk_with_perturbation = size_of_an_element

channels_position_from_zero = space
channels_end_from_zero = channels_position_from_zero + thick

xpFine = util.pCoordinates(NFine)
xtFine = util.tCoordinates(NFine)

#I want to know the exact places of the channels
ref_array = aFine_ref_shaped[0]

epsilonT = []

forward_mapping = np.stack([xpFine[:, 0], xpFine[:, 1]], axis=1)

xpFine_shaped = xpFine.reshape(fine + 1, fine + 1, 2)

for i in [1,3]:

    middle = int((fine + 1) * (i / 4))
    intervall = int((fine + 1) / 8)

    left = middle - intervall
    right = middle + intervall

    left, right = 0, fine + 1

    left_2 = middle - int(intervall)
    right_2 = middle + int(intervall)

    left, right = left_2, right_2

    print(fine + 1, left, right)
    # begin = int((fine+1)/4)

    part_x = xpFine_shaped[left:right, left_2:right_2, 0]
    part_y = xpFine_shaped[left:right, left_2:right_2, 1]
    left_margin_x = np.min(part_x)
    right_margin_x = np.max(part_x)
    left_margin_y = np.min(part_y)
    right_margin_y = np.max(part_y)

    print(left_margin_x, right_margin_x, left_margin_y, right_margin_y)

    epsilon = 25 / (right_margin_y - left_margin_y)  # why does this have to be so large???

    forward_mapping_partial = np.stack([xpFine_shaped[left:right, left_2:right_2, 0]
                                        + epsilon *
                                        (xpFine_shaped[left:right, left_2:right_2, 0] - left_margin_x) *
                                        (right_margin_x - xpFine_shaped[left:right, left_2:right_2, 0]) *
                                        (xpFine_shaped[left:right, left_2:right_2, 1] - left_margin_y) *
                                        (right_margin_y - xpFine_shaped[left:right, left_2:right_2, 1]),
                                        xpFine_shaped[left:right, left_2:right_2, 1]], axis=2)

    forward_mapping_shaped = forward_mapping.reshape(fine + 1, fine + 1, 2)
    forward_mapping_shaped[left:right, left_2:right_2, :] = forward_mapping_partial

forward_mapping = forward_mapping_shaped.reshape((fine + 1) ** 2, 2)

print('Those are the results of the shift epsilon', epsilonT)

psi = discrete_mapping.MappingCQ1(NFine, forward_mapping)



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

# aFine_trans is the transformed perturbed reference coefficient
aFine_trans = np.einsum('tij, t, tkj, t -> tik', psi.Jinv(xtFine), aFine_ref, psi.Jinv(xtFine), psi.detJ(xtFine))

plt.figure('transformed')
drawCoefficient_origin(NFine, aFine_trans)
plt.show()