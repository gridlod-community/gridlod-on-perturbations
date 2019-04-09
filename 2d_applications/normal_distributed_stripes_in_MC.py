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


#global variables
global aFine_ref
global aFine_trans
global aFine_pert
global k
global KmsijT
global correctorsListT

# Set reference coefficient
aFine_ref_shaped = CoefClass.BuildCoefficient()
aFine_ref_shaped = CoefClass.SpecificMove(Number=np.arange(0,10), steps=4, Right=1)
aFine_ref = aFine_ref_shaped.flatten()
number_of_channels = len(CoefClass.ShapeRemember)

# Discrete mapping
Nmapping = np.array([int(fine),int(fine)])

size_of_an_element = 1./fine
walk_with_perturbation = size_of_an_element

channels_position_from_zero = space
channels_end_from_zero = channels_position_from_zero + thick

xpFine = util.pCoordinates(NFine)
xtFine = util.tCoordinates(NFine)

NWorldCoarse = np.array([N, N])
boundaryConditions = np.array([[0, 0],[0, 0]])

NCoarseElement = NFine // NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
every_psi_was_valid = []
k = 3

#I want to know the exact places of the channels
ref_array = aFine_ref_shaped[0]

def create_psi_function():
    cq1 = np.zeros((int(fine) + 1, int(fine) + 1))
    for c in range(number_of_channels):
        count = 0
        for i in range(np.size(ref_array)):
            if ref_array[i] == 1:
                count +=1
            if count == (c+1)*thick:
                begin = i + 1 - space // 2
                end = i + 1 + thick+ space // 2
                break

        increasing_length = (end-begin)//2 - thick - 1
        constant_length = (end-begin) - increasing_length * 2
        epsilon = np.random.binomial(increasing_length-2,0.2)
        minus = random.sample([-1,1], 1)[0]
        epsilon *= minus
        #epsilon = random.sample(list(np.arange(-increasing_length+3,increasing_length-2,1)), 1)[0]
        #print(epsilon)
        maximal_walk = increasing_length * walk_with_perturbation
        walk = epsilon * walk_with_perturbation
        for i in range(increasing_length):
            cq1[:, begin+1+i] = (i+1)/increasing_length * walk
            cq1[:, begin + increasing_length + i + constant_length] = walk - (i+1)/increasing_length * walk

        for i in range(constant_length):
            cq1[:, begin + increasing_length + i] = walk

    cq1 = cq1.flatten()

    alpha = 1.

    for_mapping = np.stack((xpFine[:, 0] + alpha * func.evaluateCQ1(Nmapping, cq1, xpFine), xpFine[:, 1]), axis=1)
    psi = discrete_mapping.MappingCQ1(NFine, for_mapping)
    return psi, cq1

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
    rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine_pert)

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

def Monte_Carlo_simulation():
    print('Computing Monte Carlo step')

    global aFine_ref
    global aFine_trans
    global aFine_pert
    global k
    global KmsijT
    global correctorsListT

    aFine_ref = aFine_ref_shaped.flatten()
    psi, cq1 = create_psi_function()

    # plt.figure('domain mapping')
    # plt.plot(np.arange(0, fine + 1), cq1[0, :], label='$id(x) - \psi(x)$')
    # plt.plot(np.arange(0, fine), ref_array * 0.01)
    # plt.title('Domain mapping')
    # plt.legend()

    xpFine_pert = psi.evaluate(xpFine)
    xpFine_ref = psi.inverse_evaluate(xpFine)

    xtFine_pert = psi.evaluate(xtFine)
    xtFine_ref = psi.inverse_evaluate(xtFine)

    aFine_pert = func.evaluateDQ0(NFine, aFine_ref, xtFine_ref)
    aBack_ref = func.evaluateDQ0(NFine, aFine_pert, xtFine_pert)

    print('Psi is invertible if this is zero: {}'.format(np.linalg.norm(aBack_ref-aFine_ref)))
    every_psi_was_valid.append(np.linalg.norm(aBack_ref-aFine_ref))
    #aFine_trans is the transformed perturbed reference coefficient
    aFine_trans = np.einsum('tji, t, tkj, t -> tik', psi.Jinv(xtFine), aFine_ref, psi.Jinv(xtFine), psi.detJ(xtFine))

    f_pert = np.ones(np.prod(NFine+1))
    f_ref = func.evaluateCQ1(NFine, f_pert, xpFine_pert)
    f_trans = np.einsum('t, t -> t', f_ref, psi.detJ(xpFine))

    uFineFull_pert, AFine_pert, MFine = femsolver.solveFine(world, aFine_pert, f_pert, None, boundaryConditions)
    uFineFull_trans, AFine_trans, _ = femsolver.solveFine(world, aFine_trans, f_trans, None, boundaryConditions)

    uFineFull_trans_pert = func.evaluateCQ1(NFine, uFineFull_trans, xpFine_ref)

    energy_norm = np.sqrt(np.dot(uFineFull_pert, AFine_pert * uFineFull_pert))
    energy_error = np.sqrt(np.dot((uFineFull_trans_pert - uFineFull_pert), AFine_pert * (uFineFull_trans_pert - uFineFull_pert)))
    print("Energy norm {}, error {}, rel. error {}".format(energy_norm, energy_error, energy_error/energy_norm))

    Aeye = np.tile(np.eye(2), [np.prod(NFine), 1, 1])
    aFine_ref = np.einsum('tji, t-> tji', Aeye, aFine_ref)

    print('compute domain mapping error indicators')
    epsFine, epsCoarse = zip(*map(computeIndicators, range(world.NtCoarse)))

    print('apply tolerance')
    Elements_to_be_updated = []
    TOL = 0.1
    for i in range(world.NtCoarse):
        if epsFine[i] >= TOL:
            Elements_to_be_updated.append(i)

    print('.... to be updated for domain mapping: {}%'.format(np.size(Elements_to_be_updated) / np.size(epsFine) * 100))

    print('update correctors')
    if np.size(Elements_to_be_updated) == 0:
        correctorsListTNew, KmsijTNew = correctorsListT, KmsijT
    else:
        patchT_irrelevant, correctorsListTNew, KmsijTNew, csiTNew = zip(*map(UpdateCorrectors, Elements_to_be_updated))

    KmsijT_list = list(KmsijT)
    correctorsListT_list = list(correctorsListT)
    i = 0
    for T in Elements_to_be_updated:
        KmsijT_list[T] = KmsijTNew[i]
        correctorsListT_list[T] = correctorsListTNew[i]
        i += 1

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
    uLodFine_METHOD = uLodFine
    newErrorFine = np.sqrt(np.dot(uLodFine - uFineFull_trans, AFine_trans * (uLodFine - uFineFull_trans)))
    print('Method error: {}'.format(newErrorFine))

    print('update all correctors')
    patchT_irrelevant, correctorsListT, KmsijT, csiTNew = zip(*map(UpdateCorrectors, range(world.NtCoarse)))

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
    newErrorFine = np.sqrt(np.dot(uLodFine - uFineFull_trans, AFine_trans * (uLodFine - uFineFull_trans)))
    print('Exact LOD error: {}'.format(newErrorFine))

    return uLodFine_METHOD, uLodFine, uFineFull_pert, MFine

print('start to compute offline stage')
# Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
patchT, correctorsListT, KmsijT, csiT = zip(*map(computeKmsij, range(world.NtCoarse)))

print('Starting Monte Carlo method...')
MC = 10
uM = []
uE = []
uFEM = []
MF = []

for m in range(MC):
    print('________________ step {} ______________'.format(m))
    uLodFine_METHOD, uLodFine_exact_LOD, uAim, MFine = Monte_Carlo_simulation()
    uM.append(uLodFine_METHOD)
    uE.append(uLodFine_exact_LOD)
    uFEM.append(uAim)
print('every_psi_was_valid if this is zero: {}'.format(np.linalg.norm(every_psi_was_valid)))
print('finished')
xmLoda = np.zeros([MC])
xmVcLoda = np.zeros([MC])
xmLodVcLoda = np.zeros([MC])

ems = []

plottingx = np.zeros([MC - 1])
plottingy = np.zeros([MC - 1])
plottingz = np.zeros([MC - 1])

plotting2x = np.zeros([MC - 1])
plotting2y = np.zeros([MC - 1])
plotting2z = np.zeros([MC - 1])

plotting3x = np.zeros([MC - 1])
plotting3y = np.zeros([MC - 1])
plotting3z = np.zeros([MC - 1])

for i in range(MC):
    uVcLod = uM[i]
    uLod = uE[i]
    uFineFem = uFEM[i]
    eVcLod = np.sqrt(np.dot(uFineFem - uVcLod, MFine * (uFineFem - uVcLod))) / np.sqrt(
        np.dot(uFineFem, MFine * uFineFem))
    eLodVcLod = np.sqrt(np.dot(uVcLod - uLod, MFine * (uVcLod - uLod))) / np.sqrt(np.dot(uLod, MFine * uLod))
    eLod = np.sqrt(np.dot(uFineFem - uLod, MFine * (uFineFem - uLod))) / np.sqrt(np.dot(uFineFem, MFine * uFineFem))

    xmLoda[i] = eLod
    xmVcLoda[i] = eVcLod
    xmLodVcLoda[i] = eLodVcLod

    if i == 0:
        continue
    ems.append(i + 1)

    muLod = 0
    muVcLod = 0
    muLodVcLod = 0
    for j in range(0, i + 1):
        muLod += xmLoda[j]
        muVcLod += xmVcLoda[j]
        muLodVcLod += xmLodVcLoda[j]

    muLod /= i + 1
    muVcLod /= i + 1
    muLodVcLod /= i + 1

    sig2Lod = 0
    sig2VcLod = 0
    sig2LodVcLod = 0

    for j in range(0, i + 1):
        sig2Lod += (xmLoda[j] - muLod) ** (2)
        sig2VcLod += (xmVcLoda[j] - muVcLod) ** (2)
        sig2LodVcLod += (xmLodVcLoda[j] - muLodVcLod) ** (2)

    sig2Lod /= i
    sig2VcLod /= i
    sig2LodVcLod /= i

    a = [np.sqrt(sig2Lod) / np.sqrt(i + 1) * 1.96, np.sqrt(sig2VcLod) / np.sqrt(i + 1) * 1.96,
         np.sqrt(sig2LodVcLod) / np.sqrt(i + 1) * 1.96]
    mum = [muLod, muVcLod, muLodVcLod]

    plottingx[i - 1] = mum[0] - a[0]
    plottingy[i - 1] = mum[0]
    plottingz[i - 1] = mum[0] + a[0]

    plotting2x[i - 1] = mum[1] - a[1]
    plotting2y[i - 1] = mum[1]
    plotting2z[i - 1] = mum[1] + a[1]

    plotting3x[i - 1] = mum[2] - a[2]
    plotting3y[i - 1] = mum[2]
    plotting3z[i - 1] = mum[2] + a[2]

# fig = plt.figure('Monte Carlo result')
# ax = fig.add_subplot(111)
# ax.set_title('Monte Carlo')
# ax.semilogy(ems,plotting2y, linewidth = 0.8)
# ax.semilogy(ems,plotting2x, linewidth = 0.8)
# ax.semilogy(ems,plotting2z, linewidth = 0.8)
#
# ax.semilogy(ems,plottingy, '--', linewidth = 0.8)
# ax.semilogy(ems,plottingx, '--', linewidth = 0.8)
# ax.semilogy(ems,plottingz, '--', linewidth = 0.8)


plt.show()