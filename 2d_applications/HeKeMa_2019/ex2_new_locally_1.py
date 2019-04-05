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
from gridlod_on_perturbations.visualization_tools import d3sol, drawCoefficient_origin
from MasterthesisLOD.visualize import drawCoefficientGrid, drawCoefficient
import csv

fine = 256
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)

space = 20
thick = 3

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

# CoefClassRhs = buildcoef2d.Coefficient2d(NFine+1,
#                         bg                  = 0.,
#                         val                 = val,
#                         length              = 1,
#                         thick               = thick+2,
#                         space               = space,
#                         probfactor          = 1,
#                         right               = 1,
#                         down                = 0,
#                         diagr1              = 0,
#                         diagr2              = 0,
#                         diagl1              = 0,
#                         diagl2              = 0,
#                         LenSwitch           = None,
#                         thickSwitch         = None,
#                         equidistant         = True,
#                         ChannelHorizontal   = None,
#                         ChannelVertical     = True,
#                         BoundarySpace       = True)

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

f_pert = np.ones(NpFine)
#f_pert = np.block([[np.zeros((fine,1)), aFine_ref_shaped], [np.zeros((1,fine+1))]]).flatten()

# f_pert = CoefClassRhs.BuildCoefficient().flatten()
# plt.figure('localized right hand side')
# drawCoefficient_origin(NFine+1, f_pert)

xpFine = util.pCoordinates(NFine)
xtFine = util.tCoordinates(NFine)

def create_psi_function():
    global aFine_pert
    global f_trans
    forward_mapping = np.stack([xpFine[:, 0], xpFine[:, 1]], axis=1)

    xpFine_shaped = xpFine.reshape(fine + 1, fine + 1, 2)

    middle = int((fine + 1) / 2)
    intervall = int((fine + 1) / 8) - 12

    left = middle - intervall
    right = middle + intervall

    left, right = 0, fine + 1

    left_2 = middle - int(intervall) - 12
    right_2 = middle + int(intervall) - 12

    # left , right = left_2, right_2

    print(fine + 1, left, right)
    # begin = int((fine+1)/4)

    part_x = xpFine_shaped[left:right, left_2:right_2, 0]
    part_y = xpFine_shaped[left:right, left_2:right_2, 1]
    left_margin_x = np.min(part_x)
    right_margin_x = np.max(part_x)
    left_margin_y = np.min(part_y)
    right_margin_y = np.max(part_y)

    print(left_margin_x, right_margin_x, left_margin_y, right_margin_y)

    epsilon = 20  # why does this have to be so large???

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

    psi = discrete_mapping.MappingCQ1(NFine, forward_mapping)
    return psi

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
ROOT = '../results/local_bending_stripes_1/'



eps_ranges = [1]
MC = 1
NList = [32]
kList = [2]

# TOL = 100
# for tol in range(20):
#     TOL.append(np.round((10 - tol / 2) * 10, 0))
#     TOL.append(np.round((10 - tol / 2) * 1, 1))
#     TOL.append(np.round((10 - tol / 2) * 0.1, 2))
#     TOL.append(np.round((10 - tol / 2) * 0.01, 3))
#     TOL.append(np.round((10 - tol / 2) * 0.001, 4))
#     TOL.append(np.round((10 - tol / 2) * 0.0001, 5))
#     TOL.append(np.round((10 - tol / 2) * 0.00001, 6))
# TOL = list(set(TOL))
# TOL.sort()
# TOL = TOL[len(TOL) - 1:0:-1]

for eps_range in eps_ranges:
    for m in range(MC):
        # print('________________ step {} ______________'.format(m), end='')
        psi = create_psi_function()
        plt.figure("Coefficient")
        drawCoefficient_origin(NFine, aFine_ref)

        plt.figure("a_perturbed")
        drawCoefficient_origin(NFine, aFine_pert)

        for k in kList:
            for N in NList:
                print('precomputing for k {} and N {}  ...... '.format(k, N))
                TOL = 100
                TOLt = []
                to_be_updatedT_DM = []
                to_be_updatedT_CL = []
                energy_errorT = []
                tmp_errorT = []

                NWorldCoarse = np.array([N, N])
                boundaryConditions = np.array([[0, 0], [0, 0]])

                NCoarseElement = NFine // NWorldCoarse
                world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
                xpFine_ref = psi.inverse_evaluate(xpFine)

                patchT, correctorsListT, KmsijT, csiT = zip(*map(computeKmsij, range(world.NtCoarse)))

                epsFine_DM, epsFine_CL = Monte_Carlo_recomputations(psi)
                already_updated = []

                fig = plt.figure("error indicator k {} and N {}".format(k,N))
                ax = fig.add_subplot(1, 1, 1)
                ax.set_title("error indicator k {} and N {}".format(k,N))
                np_eps = np.einsum('i,i -> i', np.ones(np.size(epsFine_DM)), epsFine_DM)
                drawCoefficientGrid(NWorldCoarse, np_eps, fig, ax, original_style=True)

                uFineFull_pert, AFine_pert, _ = femsolver.solveFine(world, aFine_pert, f_pert, None, boundaryConditions)
                uFineFull_trans, AFine_trans, _ = femsolver.solveFine(world, aFine_trans, f_trans, None,
                                                                      boundaryConditions)

                uFineFull_trans_pert = func.evaluateCQ1(NFine, uFineFull_trans, xpFine_ref)

                energy_norm = np.sqrt(np.dot(uFineFull_pert, AFine_pert * uFineFull_pert))
                energy_error = np.sqrt(np.dot((uFineFull_trans_pert - uFineFull_pert),
                                              AFine_pert * (uFineFull_trans_pert - uFineFull_pert)))
                print("Energy norm {}, error {}, rel. error {}".format(energy_norm, energy_error,
                                                                       energy_error / energy_norm))

                print('k: {}, N: {}'.format(k, N))

                uFineFull_pert, AFine_pert = uFineFull_trans, AFine_trans
                init = 1

                epsFine_DM = {i: epsFine_DM[i] for i in range(np.size(epsFine_DM)) if epsFine_DM[i] > 0}
                epsFine_CL = {i: epsFine_CL[i] for i in range(np.size(epsFine_CL)) if epsFine_CL[i] > 0}

                print('Starting Algorithm ...... ')

                #print('length of epsFine ', len(epsFine_DM))
                continue_computing = 1
                while continue_computing:
                    TOLt.append(TOL)
                    Elements_to_be_updated_DM = []
                    Elements_to_be_updated_CL = []
                    for (i,eps) in epsFine_DM.items():
                        if eps >= TOL:
                            if i not in already_updated:
                                already_updated.append(i)
                                Elements_to_be_updated_DM.append(i)

                    to_be_updated_DM = np.size(already_updated) / len(epsFine_DM) * 100
                    to_be_updatedT_DM.append(to_be_updated_DM)
                    for (i, eps) in epsFine_CL.items():
                        if eps >= TOL:
                            Elements_to_be_updated_CL.append(i)
                    to_be_updated_CL = np.size(Elements_to_be_updated_CL) / len(epsFine_CL) * 100
                    to_be_updatedT_CL.append(to_be_updated_CL)

                    ## update domain mapping
                    if np.size(Elements_to_be_updated_DM) is not 0:
                        #print('.... to_be_updated_DM for tol {} : {}'.format(tol, to_be_updated_DM))
                        #print('update correctors')
                        _ , correctorsListTNew, KmsijTNew, _ = zip(
                            *map(UpdateCorrectors, Elements_to_be_updated_DM))
                    else:
                        if init:
                            #print('.... to_be_updated_DM for tol {} : {}'.format(tol, to_be_updated_DM))
                            pass
                        else:
                            energy_errorT.append(energy_error)
                            tmp_errorT.append(old_tmp_energy_error)
                            if np.size(already_updated) / len(epsFine_DM)==1:
                                print('     every corrector has been updated')
                                continue_computing = 0
                                continue
                            else:
                                print('     skipping TOL {}'.format(TOL))
                                TOL/=2.
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

                    uFineFull_trans_pert = uLodFine

                    if init:
                        uFineFull_trans_pert_old = uLodFine
                        init = 0

                    #tmp_error
                    tmp_energy_error = np.sqrt(
                        np.dot((uFineFull_trans_pert - uFineFull_trans_pert_old), AFine_pert * (uFineFull_trans_pert - uFineFull_trans_pert_old)))
                    old_tmp_energy_error = tmp_energy_error
                    #actual error
                    energy_error = np.sqrt(
                        np.dot((uFineFull_trans_pert - uFineFull_pert), AFine_pert * (uFineFull_trans_pert - uFineFull_pert)))

                    uFineFull_trans_pert_old = uFineFull_trans_pert

                    print('          TOL: {}, updates: {}%, energy error: {}, tmp_error:{}'.format(TOL,to_be_updated_DM,energy_error, tmp_energy_error))
                    energy_errorT.append(energy_error)
                    tmp_errorT.append(tmp_energy_error)
                    if tmp_energy_error > 0.0001:
                        TOL /= 2.
                    else:
                        if int(np.size(already_updated) / len(epsFine_DM)) == 1:
                            if np.size(Elements_to_be_updated_DM) is not 0:
                                print('     stop computing')
                                continue_computing = 0

                plt.figure('average epsilon = ' + str(eps_range))
                plt.title('average' + str(eps_range))
                plt.semilogx(TOLt, to_be_updatedT_DM, label='domain mapping')
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
                with open('{}/{}_k{}_H{}_tmp_error.txt'.format(ROOT, eps_range, k, N), 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    for val in tmp_errorT:
                        writer.writerow([val])
                with open('{}/TOLs_k{}_H{}.txt'.format(ROOT,k,N), 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    for val in TOLt:
                        writer.writerow([val])




plt.show()