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

bg = 0.1  # background
val = 1  # values
space = 4
thick = 4
length = 4
fine = 256
NFine = np.array([fine, fine])

#With this array, I construct the coefficient. It is a new feature in buildcoef2d
ChoosingShapes = np.array([
    # shape , len, thick, space
    [   1,      4,     4,   4],
    [1, 4, 4, 4],
    [1, 4, 4, 4],
    [1, 4, 4, 4],
    [1, 4, 4, 4],
    [1, 4, 4, 4],
    [1, 4, 4, 4],
    [1, 4, 4, 4],
    [1, 4, 4, 4],
    [1, 4, 4, 4],
    [4, fine - 8, 4, 20],
    [4, fine - 8, 4, 20],
    [4, fine - 8, 4, 20],
    [4, fine - 8, 4, 20]])

CoefClass = buildcoef2d.Coefficient2d(NFine,
                                      bg=bg,
                                      val=val,
                                      right=1,
                                      thick=thick,
                                      space=space,
                                      length=length,
                                      LenSwitch=None,
                                      thickSwitch=None,
                                      equidistant=True,
                                      ChannelHorizontal=None,
                                      ChannelVertical=None,
                                      BoundarySpace=True,
                                      probfactor=1,
                                      ChoosingShapes=ChoosingShapes)

A = CoefClass.BuildCoefficient()

#But for now, the coefficient class makes a small mistake, thus I let the fails disappear.
Number = [8, 9]
Correct = CoefClass.SpecificVanish(Number=Number)

#Check whether the coefficient is correct
plt.figure("Coefficient_")
drawCoefficient(NFine, Correct.flatten())

# This is for adding defects. If you want defects you have to uncomment line 77 here
random.seed(32)
lis = np.zeros(80)
lis[0] = 1
for i in range(np.shape(CoefClass.ShapeRemember)[0]):
    Number.append(i * random.sample(list(lis), 1)[0])
Perturbed = CoefClass.SpecificVanish(Number=Number, Original=True).flatten()
# Perturbed = Correct.flatten()

# basic init
aFine_ref_shaped = Correct
aFine_ref = aFine_ref_shaped.flatten()

#Now I construct the psi with DG functions

number_of_perturbed_channels = 4
#I want to know the exact places of the channels
ref_array = aFine_ref_shaped[4]
now = 0
count = 0
for i in range(np.size(ref_array)):
    if ref_array[i] == 1:
        count +=1
    if count == 8 * thick:   #at the 8ths shape (which is the last dot in one line, the cq starts)
        begin = i+1
        break
count = 0
for i in range(np.size(ref_array)):
    if ref_array[i] == 1:
        count +=1
    if count == 13 * thick - 3 :  #it ends after the last channel
        end = i
        break

# Discrete mapping
Nmapping = np.array([int(fine),int(fine)])
cq1 = np.zeros((int(fine)+1,int(fine)+1))

# I only want to perturb on the fine mesh.
size_of_an_element = 1./fine
walk_with_perturbation = size_of_an_element

channels_position_from_zero = space
channels_end_from_zero = channels_position_from_zero + thick

#The next only have the purpose to make the psi invertible.
left = begin
right = end
increasing_length = (end-begin)//(number_of_perturbed_channels + 1) - thick -2
constant_length = (end-begin) - increasing_length * 2
maximum_walk = (increasing_length-6) * walk_with_perturbation
walk_with_perturbation = maximum_walk
for i in range(increasing_length):
    cq1[:, begin+1+i] = (i+1)/increasing_length * walk_with_perturbation
    cq1[:, begin + increasing_length + i + constant_length] = walk_with_perturbation - (i+1)/increasing_length * walk_with_perturbation
for i in range(constant_length):
    cq1[:, begin + increasing_length + i] = walk_with_perturbation

#Check what purtubation I have
plt.figure('DomainMapping')
plt.plot(np.arange(0,fine+1),cq1[space,:], label= '$id(x) - \psi(x)$')
plt.plot(np.arange(0,fine),aFine_ref_shaped[space,:], label= '$aFine$')
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
aFine_pert = func.evaluateDQ0(NFine, Perturbed, xtFine_ref)
aBack_ref = func.evaluateDQ0(NFine, aFine_pert, xtFine_pert)

plt.figure("Coefficient")
drawCoefficient(NFine, aFine_ref)

plt.figure("a_perturbed")
drawCoefficient(NFine, aFine_pert)

plt.figure("a_back")
drawCoefficient(NFine, aBack_ref)

plt.figure("Perturbation with defects")
Perturbed_and_shifted = func.evaluateDQ0(NFine, Perturbed, xtFine_ref)
drawCoefficient(NFine, Perturbed_and_shifted)

plt.show()