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

bg = 0.01  # background
val = 1  # values
space = 1
thick = 1
fine = 64
NFine = np.array([fine, fine])

#With this array, I construct the coefficient. It is a new feature in buildcoef2d
ChoosingShapes = np.array([
    # shape , len, thick, space
    [   1,      1,     1,   1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [4, fine - 2, 1, 5],
    [4, fine - 2, 1, 5],
    [4, fine - 2, 1, 5],
    [4, fine - 2, 1, 5]])

CoefClass = buildcoef2d.Coefficient2d(NFine,
                                      bg=bg,
                                      val=val,
                                      right=1,
                                      thick=thick,
                                      space=space,
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

# This is for adding defects. If you want defects you have to uncomment the last line here
random.seed(32)
lis = np.zeros(80)
lis[0] = 1
for i in range(np.shape(CoefClass.ShapeRemember)[0]):
    Number.append(i * random.sample(list(lis), 1)[0])
Perturbed = CoefClass.SpecificVanish(Number=Number, Original=True).flatten()
Perturbed = Correct.flatten()

# basic init
aFine_ref_shaped = Correct
aFine_ref = aFine_ref_shaped.flatten()

#Now I construct the psi with DG functions

number_of_perturbed_channels = 4
#I want to know the exact places of the channels
ref_array = aFine_ref_shaped[1]
now = 0
count = 0
for i in range(np.size(ref_array)):
    if ref_array[i] == 1:
        count +=1
    if count == 8:   #at the 8ths shape (which is the last dot in one line, the cq starts)
        begin = i+1
        break
count = 0
for i in range(np.size(ref_array)):
    if ref_array[i] == 1:
        count +=1
    if count == 13:  #it ends after the last channel
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
increasing_length = (end-begin)//number_of_perturbed_channels - thick - 1 -1
constant_length = (end-begin) - increasing_length * 2
maximum_walk = (increasing_length-2) * walk_with_perturbation
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

# aFine_trans is the transformed perturbed reference coefficient
aFine_trans = np.einsum('tji, t, tkj, t -> tik', psi.Jinv(xtFine), Perturbed, psi.Jinv(xtFine), psi.detJ(xtFine))

f_pert = np.ones(np.prod(NFine+1))
f_ref = func.evaluateCQ1(NFine, f_pert, xpFine_pert)
f_trans = np.einsum('t, t -> t', f_ref, psi.detJ(xpFine))

#d3sol(NFine,f, 'right hand side NT')
d3sol(NFine, f_trans, 'right hand side T')

N = 16
NWorldCoarse = np.array([N, N])
boundaryConditions = np.array([[0, 0],[0, 0]])

NCoarseElement = NFine // NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

# Naming of solutions
# ._pert        is a solution in the perturbed domain
# ._trans       is a solution in the reference domain, after transformation
# ._trans_pert  is a solution in the perturbed domain, solved in the reference domain after transformation,
#               but then remapped to the perturbed domain
uFineFull_pert, AFine_pert, _ = femsolver.solveFine(world, aFine_pert, f_pert, None, boundaryConditions)
uFineFull_trans, AFine_trans, _ = femsolver.solveFine(world, aFine_trans, f_trans, None, boundaryConditions)

uFineFull_trans_pert = func.evaluateCQ1(NFine, uFineFull_trans, xpFine_ref)

energy_norm = np.sqrt(np.dot(uFineFull_pert, AFine_pert * uFineFull_pert))
energy_error = np.sqrt(np.dot((uFineFull_trans_pert - uFineFull_pert), AFine_pert * (uFineFull_trans_pert - uFineFull_pert)))
print("Energy norm {}, error {}, rel. error {}".format(energy_norm, energy_error, energy_error/energy_norm))


'''
Plot solutions
'''
fig = plt.figure(str(N))
#fig.subplots_adjust(left=0.01,bottom=0.04,right=0.99,top=0.95,wspace=0,hspace=0.2)

ax = fig.add_subplot(221)
ax.set_title('Solution to perturbed problem (perturbed domain)',fontsize=6)
ax.imshow(np.reshape(uFineFull_pert, NFine+1), origin='lower_left')

ax = fig.add_subplot(222)
ax.set_title('Solution to transformed problem (reference domain)',fontsize=6)
ax.imshow(np.reshape(uFineFull_trans, NFine+1), origin='lower_left')

ax = fig.add_subplot(223)
ax.set_title('Solution to remapped transformed problem (perturbed domain)',fontsize=6)
ax.imshow(np.reshape(uFineFull_trans_pert, NFine+1), origin='lower_left')

ax = fig.add_subplot(224)
ax.set_title('Absolute error between perturbed and remapped transformed',fontsize=6)
im = ax.imshow(np.reshape(uFineFull_trans_pert - uFineFull_pert, NFine+1), origin='lower_left')
fig.colorbar(im)

k = 3

Aeye = np.tile(np.eye(2), [np.prod(NFine), 1, 1])
aFine_ref = np.einsum('tji, t-> tji', Aeye, aFine_ref)

def computeKmsij(TInd):
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    aPatch = lambda: coef.localizeCoefficient(patch, aFine_ref)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

def computeIndicators(TInd):
    aPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine_ref)
    rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine_trans)

    epsFine = lod.computeBasisErrorIndicatorFine(patchT[TInd], correctorsListT[TInd], aPatch, rPatch)
    epsCoarse = 0
    #epsCoarse = lod.computeErrorIndicatorCoarseFromCoefficients(patchT[TInd], csiT[TInd].muTPrime,  aPatch, rPatch)
    return epsFine, epsCoarse

def computeIndicators_classic(TInd):
    aPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine_ref_shaped.flatten())
    rPatch = lambda: coef.localizeCoefficient(patchT[TInd], Perturbed_and_shifted)

    epsFine = lod.computeBasisErrorIndicatorFine(patchT[TInd], correctorsListT[TInd], aPatch, rPatch)
    epsCoarse = 0
    #epsCoarse = lod.computeErrorIndicatorCoarseFromCoefficients(patchT[TInd], csiT[TInd].muTPrime,  aPatch, rPatch)
    return epsFine, epsCoarse

def UpdateCorrectors(TInd):
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine_trans)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, rPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, rPatch)
    return patch, correctorsList, csi.Kmsij, csi


# Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
patchT, correctorsListT, KmsijT, csiT = zip(*map(computeKmsij, range(world.NtCoarse)))

print('compute domain mapping error indicators')
epsFine, epsCoarse = zip(*map(computeIndicators, range(world.NtCoarse)))

fig = plt.figure("error indicator")
ax = fig.add_subplot(1,1,1)
ax.set_title("Error indicator in the reference domain")
np_eps = np.einsum('i,i -> i', np.ones(np.size(epsFine)), epsFine)
drawCoefficientGrid(NWorldCoarse, np_eps,fig,ax, original_style=True, logplot=True)

print('compute classic error indicators')
epsFine_classic, epsCoarse = zip(*map(computeIndicators_classic, range(world.NtCoarse)))

fig = plt.figure("error indicator classic")
ax = fig.add_subplot(1,1,1)
ax.set_title("Classic error indicator in the perturbed domain")
np_eps_classic = np.einsum('i,i -> i', np.ones(np.size(epsFine)), epsFine_classic)
drawCoefficientGrid(NWorldCoarse, np_eps_classic,fig,ax, original_style=True, logplot=True)

plt.figure("compare error indicators")
plt.title("Cut section of the non constant dimension")
plt.plot(np.arange(N), np_eps_classic.reshape(NWorldCoarse,order='F').T[0], label='perturbed domain')
plt.plot(np.arange(N), np_eps.reshape(NWorldCoarse,order='F').T[0], label='reference domain')
plt.legend()
plt.xlabel('fine elements')

print('apply tolerance')
Elements_to_be_updated_classic = []
Elements_to_be_updated = []
TOL = 0.1
for i in range(world.NtCoarse):
    if epsFine[i] >= TOL:
        Elements_to_be_updated.append(i)
    if epsFine_classic[i] >= TOL:
        Elements_to_be_updated_classic.append(i)

print('.... to be updated for classic: {}%'.format(np.size(Elements_to_be_updated_classic)/np.size(epsFine)*100))
print('.... to be updated for domain mapping: {}%'.format(np.size(Elements_to_be_updated)/np.size(epsFine)*100))

print('update correctors')
patchT_irrelevant, correctorsListTNew, KmsijTNew, csiTNew = zip(*map(UpdateCorrectors, Elements_to_be_updated))

print('replace Kmsij and update correctorsListT')
KmsijT_list = list(KmsijT)
correctorsListT_list = list(correctorsListT)
i=0
for T in Elements_to_be_updated:
    KmsijT_list[T] = KmsijTNew[i]
    correctorsListT_list[T] = correctorsListTNew[i]
    i+=1

KmsijT = tuple(KmsijT_list)
correctorsListT = tuple(correctorsListT_list)

print('solve the system')
KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)

MFull = fem.assemblePatchMatrix(NFine, world.MLocFine)

basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
modifiedBasis = basis - basisCorrectors

bFull = MFull * f_trans
bFull = basis.T * bFull

uFull, _ = pglod.solve(world, KFull, bFull, boundaryConditions)

uLodFine = modifiedBasis * uFull

fig = plt.figure('new figure')
ax = fig.add_subplot(121)
ax.set_title('PGLOD Solution to transformed problem (reference domain)',fontsize=6)
im = ax.imshow(np.reshape(uLodFine, NFine+1), origin='lower_left')
#fig.colorbar(im)
ax = fig.add_subplot(122)
ax.set_title('FEM Solution to transformed problem (reference domain)',fontsize=6)
im = ax.imshow(np.reshape(uFineFull_trans, NFine+1), origin='lower_left')
#fig.colorbar(im)

newErrorFine = np.sqrt(np.dot(uLodFine - uFineFull_trans, AFine_trans * (uLodFine - uFineFull_trans)))

print('Error: {}'.format(newErrorFine))

print('finished')

plt.show()