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
from gridlod_on_perturbations.visualization_tools import drawCoefficient_origin
from MasterthesisLOD.visualize import drawCoefficientGrid
import csv

import ipyparallel as ipp

client = ipp.Client(sshserver='local')
client[:].use_cloudpickle()
view = client.load_balanced_view()

factor = 2**0
fine = 256 * factor
NFine = np.array([fine,fine])
NpFine = np.prod(NFine + 1)

space = 32 * factor
thick = 4 * factor

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


# Discrete mapping
Nmapping = np.array([int(fine),int(fine)])

size_of_an_element = 1./fine
print('the size of a fine element is {}'.format(size_of_an_element))
walk_with_perturbation = size_of_an_element

channels_position_from_zero = space
channels_end_from_zero = channels_position_from_zero + thick

xpFine = util.pCoordinates(NFine)
xtFine = util.tCoordinates(NFine)

#I want to know the exact places of the channels
ref_array = aFine_ref_shaped[0]

def create_psi_function(eps_range_in):
    global aFine_pert
    epsilonT = []
    cq1 = np.zeros((int(fine) + 1, int(fine) + 1))

    cs = np.random.randint(0,2,number_of_channels)
    cs = [c * random.sample([-1,1],1)[0] for c in cs]

    # this time manually
    cs[3] = 10
    cs[4] = 2
    cs[5] = 1

    print(cs)

    last_value = 0
    for i, c in enumerate(cs):
        platform = space//2 + 2 * thick
        begin = platform//2 + i * (space + thick)
        end = begin + space - platform + thick

        epsilon = c * walk_with_perturbation
        epsilonT.append(epsilon)
        walk = epsilon - last_value

        constant_length = platform + thick
        increasing_length = end - begin

        for i in range(increasing_length):
            cq1[:, begin+i] = last_value + (i+1)/increasing_length * walk

        for i in range(constant_length):
            cq1[:, begin + increasing_length + i] = epsilon

        last_value = epsilon

    # ending
    begin += space + thick
    end = begin + space - platform + thick
    epsilon = 0
    walk = epsilon - last_value
    increasing_length = end - begin
    for i in range(increasing_length):
        cq1[:, begin+i] = last_value + (i+1)/increasing_length * walk


    plt.plot(np.arange(0, fine + 1), cq1[space, :], label='$id(x) - \psi(x)$')
    plt.title('Domain mapping')
    plt.legend()
    plt.show()

    print('These are the results of the shift epsilon', epsilonT)
    cq1 = cq1.flatten()

    alpha = 1.

    for_mapping = np.stack((xpFine[:, 0] + alpha * func.evaluateCQ1(Nmapping, cq1, xpFine), xpFine[:, 1]), axis=1)
    psi = discrete_mapping.MappingCQ1(NFine, for_mapping)

    aFine_ref = aFine_ref_shaped.flatten()

    xtFine_pert = psi.evaluate(xtFine)
    xtFine_ref = psi.inverse_evaluate(xtFine)

    aFine_pert = func.evaluateDQ0(NFine, aFine_ref, xtFine_ref)
    aBack_ref = func.evaluateDQ0(NFine, aFine_pert, xtFine_pert)

    is_this_invertible = np.linalg.norm(aBack_ref-aFine_ref)
    #print('Psi is invertible if this is zero: {}'.format(is_this_invertible))

    if is_this_invertible > 0.00001:
        print('.'.format(is_this_invertible), end='')
        return create_psi_function(eps_range)   ## make sure that it works
    else:
        print('Psi is invertible')
        return psi, cq1, epsilonT


# Functions

def computeKmsij(TInd):
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    aPatch = lambda: coef.localizeCoefficient(patch, aFine_ref)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

def computeRhsij(TInd):
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    aPatch = lambda: coef.localizeCoefficient(patch, aFine_ref)
    MPatch = coef.localizeCoefficient(patch, Mrhs)

    correctorsRhsList = lod.computeElementCorrector(patch, IPatch, aPatch, None, MPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsRhsList, aPatch, rhs = True)
    return patch, correctorsRhsList, csi.Kmsij, csi

def computeIndicators(TInd):
    aPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine_ref)
    rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine_trans)

    epsCoarse = lod.computeErrorIndicatorCoarseFromCoefficients(patchT[TInd], csiT[TInd].muTPrime,  aPatch, rPatch)
    return epsCoarse, 0

def computeIndicators_classic(TInd):
    global aFine_pert
    aPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine_ref_shaped.flatten())
    rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine_pert)

    epsCoarse = lod.computeErrorIndicatorCoarseFromCoefficients(patchT[TInd], csiT[TInd].muTPrime,  aPatch, rPatch)
    return epsCoarse, 0

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
    aFine_trans = np.einsum('tij, t, tkj, t -> tik', psi.Jinv(xtFine), aFine_ref, psi.Jinv(xtFine), psi.detJ(xtFine))

    plt.figure('transformed')
    drawCoefficient_origin(NFine, aFine_trans)
    plt.show()

    Aeye = np.tile(np.eye(2), [np.prod(NFine), 1, 1])
    aFine_ref = np.einsum('tji, t-> tji', Aeye, aFine_ref)

    epsCoarse_dom_mapping, _ = zip(*map(computeIndicators, range(world.NtCoarse)))
    epsCoarse_CLassic, _ = zip(*map(computeIndicators_classic, range(world.NtCoarse)))

    return epsCoarse_dom_mapping, epsCoarse_CLassic


#### MAIN PROGRAM

print('start to compute offline stage')
ROOT = '../results/moving_stripes/'


# This is useful for MC. For now we set MC = 1 which means only one MC computation
eps_ranges = [0]
MC = 1
NList = [32]
kList = [2]

for eps_range in eps_ranges:
    for m in range(MC):
        # print('________________ step {} ______________'.format(m), end='')
        
        # construct psi 
        psi, _, epsilon = create_psi_function(eps_range)
        
        plt.figure("Coefficient")
        drawCoefficient_origin(NFine, aFine_ref)

        plt.figure("a_perturbed")
        drawCoefficient_origin(NFine, aFine_pert)

        plt.show()
        
        for k in kList:
            for N in NList:
                print('precomputing for k {} and N {}  ...... '.format(k, N))
                TOL = 200
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

                print('compute reference correctors')
                patchT, correctorsListT, KmsijT, csiT = zip(*map(computeKmsij, range(world.NtCoarse)))

                print('compute error indicators')
                epsCoarse_DM, epsCoarse_CL = Monte_Carlo_recomputations(psi)
                already_updated = []

                fig = plt.figure("error indicator k {} and N {}".format(k, N))
                ax = fig.add_subplot(1, 1, 1)
                ax.set_title("error indicator k {} and N {}".format(k, N))
                np_eps = np.einsum('i,i -> i', np.ones(np.size(epsCoarse_DM)), epsCoarse_DM)
                drawCoefficientGrid(NWorldCoarse, np_eps, fig, ax, original_style=True)

                with open('{}/{}_k{}_H{}_eps_fine.txt'.format(ROOT, eps_range, k, N), 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    for val in epsCoarse_DM:
                        writer.writerow([val])

                xpFine_pert = psi.evaluate(xpFine)
                xpFine_ref = psi.inverse_evaluate(xpFine)
                
                # Now we localize f dependent on epsilon
                f_cheat = np.ones(np.prod(NWorldCoarse)) * bg
                worst_coarse_elements = []

                for i, eps in enumerate(epsCoarse_DM):
                    if eps >= 1.5:
                        worst_coarse_elements.append(i)

                for i in worst_coarse_elements:
                    f_cheat[i+1] = 1000.

                plt.figure('check coarse f')
                drawCoefficient_origin(NWorldCoarse, f_cheat)
                plt.show()

                f_cheat = f_cheat.reshape(N,N)
                f_cheat = np.append(f_cheat, np.ones((N, 1)) * bg, 1)
                f_row = np.ones((1,N+1))
                f_row[0] = f_cheat[0,:]
                f_cheat = np.append(f_cheat, f_row, 0)
                f_cheat = f_cheat.flatten()

                basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
                f_cheat = basis * f_cheat

                plt.figure('check fine f')
                drawCoefficient_origin(NFine+1, f_cheat)
                plt.show()

                f_ref = f_cheat
                f_pert = func.evaluateCQ1(NFine, f_ref, xpFine_ref)

                f_trans = np.einsum('t, t -> t', f_ref, psi.detJ(xpFine))

                Mrhs = f_trans

                print('compute right hand side correctors')
                #patchT, correctorsListRhsT, RhsijT, csiT = zip(*map(computeRhsij, range(world.NtCoarse)))

                # OLD f
                # f_pert = np.ones(NpFine)
                # f_ref = func.evaluateCQ1(NFine, f_pert, xpFine_pert)
                # f_pert = func.evaluateCQ1(NFine, f_ref, xpFine_ref)
                # f_trans = np.einsum('t, t -> t', f_ref, psi.detJ(xpFine))

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

                eps_size = np.size(epsCoarse_DM)
                epsCoarse_DM = {i: epsCoarse_DM[i] for i in range(np.size(epsCoarse_DM)) if epsCoarse_DM[i] > 0}
                epsCoarse_CL = {i: epsCoarse_CL[i] for i in range(np.size(epsCoarse_CL)) if epsCoarse_CL[i] > 0}
                full_percentage = len(epsCoarse_DM) / eps_size
                print(full_percentage * 100)


                print('Starting Algorithm ...... ')

                continue_computing = 1
                while continue_computing:
                    TOLt.append(TOL)
                    Elements_to_be_updated_DM = []
                    Elements_to_be_updated_CL = []
                    for (i,eps) in epsCoarse_DM.items():
                        if eps >= TOL:
                            if i not in already_updated:
                                already_updated.append(i)
                                Elements_to_be_updated_DM.append(i)

                    to_be_updated_DM = np.size(already_updated) / len(epsCoarse_DM) * 100
                    to_be_updatedT_DM.append(to_be_updated_DM* full_percentage)
                    for (i, eps) in epsCoarse_CL.items():
                        if eps >= TOL:
                            Elements_to_be_updated_CL.append(i)
                    to_be_updated_CL = np.size(Elements_to_be_updated_CL) / len(epsCoarse_CL) * 100
                    to_be_updatedT_CL.append(to_be_updated_CL* full_percentage)

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
                            if np.size(already_updated) / len(epsCoarse_DM)==1:
                                print('     every corrector has been updated')
                                continue_computing = 0
                                continue
                            else:
                                print('     skipping TOL {}'.format(TOL))
                                TOL *= 3/4.
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

                    KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)

                    RFull = 0
                    # RFull = pglod.assembleMsStiffnessMatrix(world, patchT, RhsijT)

                    MFull = fem.assemblePatchMatrix(NFine, world.MLocFine)

                    bFull = MFull * f_trans - RFull * f_trans

                    basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
                    basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
                    modifiedBasis = basis - basisCorrectors


                    bFull = basis.T * bFull

                    uFull, _ = pglod.solve(world, KFull, bFull, boundaryConditions)

                    uLodFine = modifiedBasis * uFull

                    ### TODO
                    # uLodFine += pglod.computeCorrection(correctorsListRhsT)

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

                    print('          TOL: {}, updates: {}%, energy error: {}, tmp_error:{}'.format(TOL,to_be_updated_DM* full_percentage,energy_error, tmp_energy_error))
                    energy_errorT.append(energy_error)
                    tmp_errorT.append(tmp_energy_error)
                    if tmp_energy_error > 0.0001:
                        TOL *= 3/4.
                    else:
                        if int(np.size(already_updated) / len(epsCoarse_DM)) == 1:
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