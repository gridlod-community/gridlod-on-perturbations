# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, femsolver, interp, coef, fem, lod, pglod
from gridlod.world import World, Patch


class AdaptiveAlgorithm:
    def __init__(self, world, k, boundaryConditions, a_Fine_to_be_approximated, aFine_ref, f_trans, epsCoarse, KmsijT,
                 correctorsListT, patchT, RFull, Rf, MFull, uFineFull_trans=None, AFine_trans=None, StartingTolerance=100):
        self.world = world
        self.k = k
        self.boundaryConditions = boundaryConditions
        self.a_Fine_to_be_approximated = a_Fine_to_be_approximated
        self.aFine_ref = aFine_ref
        self.f_trans = f_trans
        self.epsCoarse = epsCoarse
        self.KmsijT = KmsijT
        self.correctorsListT = correctorsListT
        self.patchT = patchT
        self.RFull = RFull
        self.Rf = Rf
        self.MFull = MFull
        self.uFineFull_trans = uFineFull_trans
        self.AFine_trans = AFine_trans
        self.StartingTolerance = StartingTolerance
        self.init = 1

    def UpdateCorrectors(self, TInd):
        # print(" UPDATING {}".format(TInd))
        patch = Patch(self.world, self.k, TInd)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, self.boundaryConditions)
        rPatch = lambda: coef.localizeCoefficient(patch, self.a_Fine_to_be_approximated)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, rPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, rPatch)
        return patch, correctorsList, csi.Kmsij, csi

    def UpdateElements(self, tol, offset= [], Printing = False):
        print('apply tolerance') if Printing else 1
        Elements_to_be_updated = []
        for (i,eps) in self.epsCoarse.items():
            if eps > tol:
                if i not in offset:
                    offset.append(i)
                    Elements_to_be_updated.append(i)
        print('... to be updated: {}%'.format(100*np.size(Elements_to_be_updated)/len(self.epsCoarse)), end='') \
            if Printing else 1

        if np.size(Elements_to_be_updated) != 0:
            print('... update correctors') if Printing else 1
            patchT_irrelevant, correctorsListTNew, KmsijTNew, csiTNew = zip(*map(self.UpdateCorrectors,
                                                                                 Elements_to_be_updated))

            print('replace Kmsij and update correctorsListT') if Printing else 1
            KmsijT_list = list(np.copy(self.KmsijT))
            correctorsListT_list = list(np.copy(self.correctorsListT))
            i = 0
            for T in Elements_to_be_updated:
                KmsijT_list[T] = KmsijTNew[i]
                correctorsListT_list[T] = correctorsListTNew[i]
                i += 1

            self.KmsijT = tuple(KmsijT_list)
            self.correctorsListT = tuple(correctorsListT_list)

            return offset, 1 # computed
        else:
            print('... there is nothing to be updated') if Printing else 1
            return offset, 0 # not computed

    def StartAlgorithm(self):
        assert(self.init)    # only start the algorithm once

        # in case not every element is affected, the percentage would be missleading.
        eps_size = np.size(self.epsCoarse)
        self.epsCoarse = {i: self.epsCoarse[i] for i in range(np.size(self.epsCoarse)) if self.epsCoarse[i] > 0}
        full_percentage = len(self.epsCoarse) / eps_size

        world = self.world
        print('starting algorithm ...... ')

        TOL = self.StartingTolerance

        TOLt = []
        to_be_updatedT = []
        energy_errorT = []
        tmp_errorT = []

        offset = []

        continue_computing = 1
        while continue_computing:
            TOLt.append(TOL)
            offset, computed = self.UpdateElements(TOL, offset, Printing=False)

            if computed:
                pass
            else:
                if self.init:
                    pass
                else:
                    to_be_updated = np.size(offset) / len(self.epsCoarse) * 100
                    to_be_updatedT.append(to_be_updated * full_percentage)

                    energy_errorT.append(energy_error)
                    tmp_errorT.append(old_tmp_energy_error)
                    if np.size(offset) / len(self.epsCoarse) == 1:
                        print('     every corrector has been updated')
                        continue_computing = 0
                        continue
                    else:
                        print('     skipping TOL {}'.format(TOL))
                        TOL *= 3/ 4.
                        continue

            to_be_updated = np.size(offset) / len(self.epsCoarse) * 100
            to_be_updatedT.append(to_be_updated * full_percentage)

            KFull = pglod.assembleMsStiffnessMatrix(world, self.patchT, self.KmsijT)

            basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)

            bFull = basis.T * self.MFull * self.f_trans - self.RFull

            basisCorrectors = pglod.assembleBasisCorrectors(world, self.patchT, self.correctorsListT)
            modifiedBasis = basis - basisCorrectors

            uFull, _ = pglod.solve(world, KFull, bFull, self.boundaryConditions)

            uLodFine = modifiedBasis * uFull
            uLodFine += self.Rf

            uFineFull_trans_LOD = uLodFine

            if self.init:
                uFineFull_trans_LOD_old = uLodFine
                self.init = 0

            # tmp_error
            tmp_energy_error = np.sqrt(
                np.dot((uFineFull_trans_LOD - uFineFull_trans_LOD_old),
                       self.AFine_trans * (uFineFull_trans_LOD - uFineFull_trans_LOD_old)))
            old_tmp_energy_error = tmp_energy_error

            # actual error
            energy_error = np.sqrt(
                np.dot((uFineFull_trans_LOD - self.uFineFull_trans),
                       self.AFine_trans * (uFineFull_trans_LOD - self.uFineFull_trans)))

            uFineFull_trans_LOD_old = uFineFull_trans_LOD

            print('      TOL: {}, updates: {}%, energy error: {}, tmp_error:{}'.format(TOL,
                                                                                       to_be_updated * full_percentage,
                                                                                       energy_error,
                                                                                       tmp_energy_error))
            energy_errorT.append(energy_error)
            tmp_errorT.append(tmp_energy_error)
            if tmp_energy_error > 1e-5:
                TOL *= 3 / 4.
            else:
                if int(np.size(offset) / len(self.epsCoarse)) == 1:
                    if computed:
                        print('     stop computing')
                        continue_computing = 0

        return to_be_updatedT, energy_errorT, tmp_errorT, TOLt, uFineFull_trans_LOD


class PercentageVsErrorAlgorithm:
    def __init__(self, world, k, boundaryConditions, a_Fine_to_be_approximated, aFine_ref, f_trans, epsCoarse, KmsijT,
                 correctorsListT, patchT, RFull, Rf, MFull, uFineFull_trans, AFine_trans):
        self.world = world
        self.k = k
        self.boundaryConditions = boundaryConditions
        self.a_Fine_to_be_approximated = a_Fine_to_be_approximated
        self.aFine_ref = aFine_ref
        self.f_trans = f_trans
        self.epsCoarse = epsCoarse
        self.KmsijT = KmsijT
        self.correctorsListT = correctorsListT
        self.patchT = patchT
        self.RFull = RFull
        self.Rf = Rf
        self.MFull = MFull
        self.uFineFull_trans = uFineFull_trans
        self.AFine_trans = AFine_trans

        self.init = 1

    def UpdateCorrectors(self, TInd):
        # print(" UPDATING {}".format(TInd))
        patch = Patch(self.world, self.k, TInd)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, self.boundaryConditions)
        rPatch = lambda: coef.localizeCoefficient(patch, self.a_Fine_to_be_approximated)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, rPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, rPatch)
        return patch, correctorsList, csi.Kmsij, csi

    def UpdateNextElement(self, tol, offset= [], Printing = False):
        print('apply tolerance') if Printing else 1
        Elements_to_be_updated = []
        for (i,eps) in self.epsCoarse.items():
            if eps > tol:
                if i not in offset:
                    offset.append(i)
                    Elements_to_be_updated.append(i)
        print('... to be updated: {}%'.format(100*np.size(Elements_to_be_updated)/len(self.epsCoarse)), end='') \
            if Printing else 1

        if np.size(Elements_to_be_updated) != 0:
            # assert(np.size(Elements_to_be_updated) == 1 or np.size(Elements_to_be_updated) == 2) # sometimes we get
            print('... update correctors') if Printing else 1
            patchT_irrelevant, correctorsListTNew, KmsijTNew, csiTNew = zip(*map(self.UpdateCorrectors,
                                                                                 Elements_to_be_updated))

            print('replace Kmsij and update correctorsListT') if Printing else 1
            KmsijT_list = list(np.copy(self.KmsijT))
            correctorsListT_list = list(np.copy(self.correctorsListT))
            i = 0
            for T in Elements_to_be_updated:
                KmsijT_list[T] = KmsijTNew[i]
                correctorsListT_list[T] = correctorsListTNew[i]
                i += 1

            self.KmsijT = tuple(KmsijT_list)
            self.correctorsListT = tuple(correctorsListT_list)

            return offset
        else:
            print('... there is nothing to be updated') if Printing else 1
            return offset

    def StartAlgorithm(self):
        assert(self.init)    # only start the algorithm once

        # in case not every element is affected, the percentage would be missleading.
        eps_size = np.size(self.epsCoarse)
        self.epsCoarse = {i: self.epsCoarse[i] for i in range(np.size(self.epsCoarse)) if self.epsCoarse[i] > 0}
        list = [ v for v in self.epsCoarse.values()]
        list.append(0)
        tols = np.sort(np.unique(list))[::-1]

        full_percentage = len(self.epsCoarse) / eps_size

        world = self.world
        print('starting algorithm ...... ')

        TOLt = []
        to_be_updatedT = []
        energy_errorT = []
        tmp_errorT = []

        offset = []
        TOL = 100   # not relevant

        for i in range(np.size(tols)):
            if TOL == 0:
                pass
            else:
                TOL = tols[i]

            TOLt.append(TOL)
            offset = self.UpdateNextElement(TOL, offset, Printing=False)

            if self.init:
                to_be_updated = np.size(offset) / len(self.epsCoarse) * 100
                to_be_updatedT.append(to_be_updated)
                pass
            else:
                to_be_updated = np.size(offset) / len(self.epsCoarse) * 100
                to_be_updatedT.append(to_be_updated * full_percentage)

            KFull = pglod.assembleMsStiffnessMatrix(world, self.patchT, self.KmsijT)

            basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)

            bFull = basis.T * self.MFull * self.f_trans - self.RFull

            basisCorrectors = pglod.assembleBasisCorrectors(world, self.patchT, self.correctorsListT)
            modifiedBasis = basis - basisCorrectors

            uFull, _ = pglod.solve(world, KFull, bFull, self.boundaryConditions)

            uLodFine = modifiedBasis * uFull
            uLodFine += self.Rf

            uFineFull_trans_LOD = uLodFine

            if self.init:
                uFineFull_trans_LOD_old = uLodFine

            # tmp_error
            tmp_energy_error = np.sqrt(
                np.dot((uFineFull_trans_LOD - uFineFull_trans_LOD_old),
                       self.AFine_trans * (uFineFull_trans_LOD - uFineFull_trans_LOD_old)))


            # actual error
            energy_error = np.sqrt(
                np.dot((uFineFull_trans_LOD - self.uFineFull_trans),
                       self.AFine_trans * (uFineFull_trans_LOD - self.uFineFull_trans)))

            uFineFull_trans_LOD_old = uFineFull_trans_LOD

            print(' step({}/{})    TOL: {}, updates: {}%, energy error: {}, tmp_error:{}'.format(i, np.size(tols), TOL,
                                                                                       to_be_updated * full_percentage,
                                                                                       energy_error,
                                                                                       tmp_energy_error))
            energy_errorT.append(energy_error)
            tmp_errorT.append(tmp_energy_error)

            if TOL == 0:
                # stop now
                break

            if tmp_energy_error < 1e-6:
                if self.init:
                    self.init = 0
                else:
                    # This is sufficient ! but compute once again for 100%
                    print('local gain is sufficiently small... aborting')
                    TOL = 0


        return to_be_updatedT, energy_errorT, tmp_errorT, TOLt, uFineFull_trans_LOD