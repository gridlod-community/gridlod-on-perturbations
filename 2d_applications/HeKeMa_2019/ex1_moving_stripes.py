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
import csv

fine = 256
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
global f_trans

# Set reference coefficient
aFine_ref_shaped = CoefClass.BuildCoefficient()
aFine_ref_shaped = CoefClass.SpecificMove(Number=np.arange(0,10), steps=4, Right=1)
aFine_ref = aFine_ref_shaped.flatten()
number_of_channels = len(CoefClass.ShapeRemember)

f_pert = np.ones(np.prod(NFine+1))

# Discrete mapping
Nmapping = np.array([int(fine),int(fine)])

size_of_an_element = 1./fine
print(size_of_an_element)
walk_with_perturbation = size_of_an_element

channels_position_from_zero = space
channels_end_from_zero = channels_position_from_zero + thick

xpFine = util.pCoordinates(NFine)
xtFine = util.tCoordinates(NFine)

#I want to know the exact places of the channels
ref_array = aFine_ref_shaped[0]

def create_psi_function(eps_range):
    global aFine_pert
    global f_trans
    epsilonT = []
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
        epsilon = np.random.uniform(-eps_range,eps_range)
        #print(epsilon)
        epsilonT.append(epsilon)
        #epsilon = random.sample(list(np.arange(-increasing_length+3,increasing_length-2,1)), 1)[0]
        #walk = epsilon * walk_with_perturbation
        walk = epsilon
        for i in range(increasing_length):
            cq1[:, begin+1+i] = (i+1)/increasing_length * walk
            cq1[:, begin + increasing_length + i + constant_length] = walk - (i+1)/increasing_length * walk

        for i in range(constant_length):
            cq1[:, begin + increasing_length + i] = walk

    cq1 = cq1.flatten()

    alpha = 1.

    for_mapping = np.stack((xpFine[:, 0] + alpha * func.evaluateCQ1(Nmapping, cq1, xpFine), xpFine[:, 1]), axis=1)
    psi = discrete_mapping.MappingCQ1(NFine, for_mapping)

    aFine_ref = aFine_ref_shaped.flatten()

    xtFine_pert = psi.evaluate(xtFine)
    xtFine_ref = psi.inverse_evaluate(xtFine)
    xpFine_pert = psi.evaluate(xpFine)
    xpFine_ref = psi.inverse_evaluate(xpFine)

    aFine_pert = func.evaluateDQ0(NFine, aFine_ref, xtFine_ref)
    aBack_ref = func.evaluateDQ0(NFine, aFine_pert, xtFine_pert)

    f_ref = func.evaluateCQ1(NFine, f_pert, xpFine_pert)
    f_trans = np.einsum('t, t -> t', f_ref, psi.detJ(xpFine))

    is_this_invertible = np.linalg.norm(aBack_ref-aFine_ref)
    #print('Psi is invertible if this is zero: {}'.format(is_this_invertible))

    if is_this_invertible > 0.001:
        print('.'.format(is_this_invertible), end='')
        return create_psi_function(eps_range)   ## make sure that it works
    else:
        print('Psi is invertible')
        return psi, cq1, epsilonT

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

    #epsFine = lod.computeBasisErrorIndicatorFine(patchT[TInd], correctorsListT[TInd], aPatch, rPatch)
    epsCoarse = 0
    epsFine = lod.computeErrorIndicatorCoarseFromCoefficients(patchT[TInd], csiT[TInd].muTPrime,  aPatch, rPatch)
    return epsFine, epsCoarse

def computeIndicators_classic(TInd):
    global aFine_pert
    aPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine_ref_shaped.flatten())
    rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine_pert)

    #epsFine = lod.computeBasisErrorIndicatorFine(patchT[TInd], correctorsListT[TInd], aPatch, rPatch)
    epsCoarse = 0
    epsFine = lod.computeErrorIndicatorCoarseFromCoefficients(patchT[TInd], csiT[TInd].muTPrime,  aPatch, rPatch)
    return epsFine, epsCoarse

def UpdateCorrectors(TInd):
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine_trans)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, rPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, rPatch)
    return patch, correctorsList, csi.Kmsij, csi

def Monte_Carlo_recomputations(psi):
    global aFine_ref
    global aFine_trans

    aFine_ref = aFine_ref_shaped.flatten()

    #aFine_trans is the transformed perturbed reference coefficient
    aFine_trans = np.einsum('tji, t, tkj, t -> tik', psi.Jinv(xtFine), aFine_ref, psi.Jinv(xtFine), psi.detJ(xtFine))

    Aeye = np.tile(np.eye(2), [np.prod(NFine), 1, 1])
    aFine_ref = np.einsum('tji, t-> tji', Aeye, aFine_ref)

    epsFine_dom_mapping, epsCoarse = zip(*map(computeIndicators, range(world.NtCoarse)))
    epsFine_classic, epsCoarse = zip(*map(computeIndicators_classic, range(world.NtCoarse)))

    return epsFine_dom_mapping, epsFine_classic

print('start to compute offline stage')
ROOT = '../results/stripes_H_k_TOL/'


eps_ranges = [0.04]
MC = 1
NList = [4,8,16,32,64,128,256]
kList = [1,2,3]

TOL = []
for tol in range(20):
    TOL.append(np.round((10 - tol / 2) * 10, 0))
    TOL.append(np.round((10 - tol / 2) * 1, 1))
    TOL.append(np.round((10 - tol / 2) * 0.1, 2))
    TOL.append(np.round((10 - tol / 2) * 0.01, 3))
    TOL.append(np.round((10 - tol / 2) * 0.001, 4))
    TOL.append(np.round((10 - tol / 2) * 0.0001, 5))
    TOL.append(np.round((10 - tol / 2) * 0.00001, 6))
TOL = list(set(TOL))
TOL.sort()
TOL = TOL[len(TOL) - 1:0:-1]

for eps_range in eps_ranges:
    print('_______________ The eps_range is {} '.format(eps_range), end='')
    for m in range(MC):
        # print('________________ step {} ______________'.format(m), end='')
        psi, _, epsilon = create_psi_function(eps_range)

        for k in kList:
            for N in NList:
                to_be_updatedT_DM = []
                to_be_updatedT_CL = []
                energy_errorT = []

                NWorldCoarse = np.array([N, N])
                boundaryConditions = np.array([[0, 0], [0, 0]])

                NCoarseElement = NFine // NWorldCoarse
                world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

                patchT, correctorsListT, KmsijT, csiT = zip(*map(computeKmsij, range(world.NtCoarse)))

                epsFine_DM, epsFine_CL = Monte_Carlo_recomputations(psi)
                already_updated = []

                print('k: {}, N: {}'.format(k, N))

                uFineFull_pert, AFine_pert, _ = femsolver.solveFine(world, aFine_pert, f_pert, None, boundaryConditions)
                #uFineFull_trans, AFine_trans, _ = femsolver.solveFine(world, aFine_trans, f_trans, None, boundaryConditions)
                init = 0
                for tol in TOL:
                    Elements_to_be_updated_DM = []
                    Elements_to_be_updated_CL = []
                    for i in range(world.NtCoarse):
                        if epsFine_DM[i] >= tol:
                            if i not in already_updated:
                                already_updated.append(i)
                                Elements_to_be_updated_DM.append(i)

                    to_be_updated_DM = np.size(already_updated) / np.size(epsFine_DM) * 100
                    to_be_updatedT_DM.append(to_be_updated_DM)
                    for i in range(world.NtCoarse):
                        if epsFine_CL[i] >= tol:
                            Elements_to_be_updated_CL.append(i)
                    to_be_updated_CL = np.size(Elements_to_be_updated_CL) / np.size(epsFine_CL) * 100
                    to_be_updatedT_CL.append(to_be_updated_CL)

                    ## update domain mapping
                    if np.size(Elements_to_be_updated_DM) is not 0:
                        #print('.... to_be_updated_DM for tol {} : {}'.format(tol, to_be_updated_DM))
                        #print('update correctors')
                        _ , correctorsListTNew, KmsijTNew, _ = zip(
                            *map(UpdateCorrectors, Elements_to_be_updated_DM))
                    else:
                        if init == 0:
                            #print('.... to_be_updated_DM for tol {} : {}'.format(tol, to_be_updated_DM))
                            init = 1
                            pass
                        else:
                            energy_errorT.append(energy_error)
                            continue

                    #print('replace Kmsij and update correctorsListT')
                    KmsijT_list = list(KmsijT)
                    correctorsListT_list = list(correctorsListT)
                    i = 0
                    for T in Elements_to_be_updated_DM:
                        #print('I am updating element {}'.format(T))
                        KmsijT_list[T] = KmsijTNew[i]
                        correctorsListT_list[T] = correctorsListTNew[i]
                        i += 1

                    KmsijT = tuple(KmsijT_list)
                    correctorsListT = tuple(correctorsListT_list)

                    #print('Norm of the matrizes {}'.format(np.linalg.norm(KmsijT-KmsijT_original)))
                    KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)

                    MFull = fem.assemblePatchMatrix(NFine, world.MLocFine)

                    basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
                    basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
                    modifiedBasis = basis - basisCorrectors

                    bFull = MFull * f_trans
                    bFull = basis.T * bFull

                    uFull, _ = pglod.solve(world, KFull, bFull, boundaryConditions)

                    uLodFine = modifiedBasis * uFull

                    xpFine_ref = psi.inverse_evaluate(xpFine)
                    uFineFull_trans_pert = func.evaluateCQ1(NFine, uLodFine, xpFine_ref)

                    #energy_norm = np.sqrt(np.dot(uFineFull_pert, MFull * uFineFull_pert))
                    energy_error = np.sqrt(
                        np.dot((uFineFull_trans_pert - uFineFull_pert), AFine_pert * (uFineFull_trans_pert - uFineFull_pert)))

                    # energy_norm = np.sqrt(np.dot(uFineFull_trans, MFull * uFineFull_trans))
                    # energy_error = np.sqrt(
                    #     np.dot((uLodFine - uFineFull_trans), MFull * (uLodFine - uFineFull_trans)))

                    #print(
                    #    "TOL {}, Energy norm {}, error {}, rel. error {}".format(tol, energy_norm, energy_error, energy_error / energy_norm))

                    print('          TOL: {}, updates: {}%, energy error: {}'.format(tol,to_be_updated_DM,energy_error))
                    energy_errorT.append(energy_error)

                plt.figure('average epsilon = ' + str(eps_range))
                plt.title('average' + str(eps_range))
                plt.semilogx(TOL, to_be_updatedT_DM, label='domain mapping')
                plt.legend()
                with open('{}/{}_k{}_H{}_DM.txt'.format(ROOT,eps_range, k, N), 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    for val in to_be_updatedT_DM:
                        writer.writerow([val])
                with open('{}/{}_k{}_H{}_CL.txt'.format(ROOT, eps_range, k, N), 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    for val in to_be_updatedT_CL:
                        writer.writerow([val])
                with open('{}/{}_k{}_H{}_error.txt'.format(ROOT, eps_range, k, N), 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    for val in energy_errorT:
                        writer.writerow([val])

with open("%s/TOLs.txt" % ROOT, 'w') as csvfile:
    writer = csv.writer(csvfile)
    for val in TOL:
        writer.writerow([val])


plt.show()