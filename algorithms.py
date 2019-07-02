# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellmann
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, femsolver, interp, coef, fem, lod, pglod
from gridlod.world import World, Patch


class AdaptiveAlgorithm:
    def __init__(self, world, k, boundaryConditions, a_Fine_to_be_approximated, aFine_ref, f_trans, E_vh, KmsijT,
                 correctorsListT, patchT, RmsijT, correctorsRhsT, MFull, uFineFull_trans=None, AFine_trans=None, StartingTolerance=100):
        self.world = world
        self.k = k
        self.boundaryConditions = boundaryConditions
        self.a_Fine_to_be_approximated = a_Fine_to_be_approximated
        self.aFine_ref = aFine_ref
        self.f_trans = f_trans
        self.E_vh = E_vh
        self.KmsijT = KmsijT
        self.correctorsListT = correctorsListT
        self.patchT = patchT
        self.RmsijT = RmsijT
        self.correctorsRhsT = correctorsRhsT
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

        MRhsList = [self.f_trans[util.extractElementFine(self.world.NWorldCoarse,
                                                  self.world.NCoarseElement,
                                                  patch.iElementWorldCoarse,
                                                  extractElements=False)]];

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, rPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, rPatch)

        correctorRhs = lod.computeElementCorrector(patch, IPatch, rPatch, None, MRhsList)[0]
        Rmsij = lod.computeRhsCoarseQuantities(patch, correctorRhs, rPatch)

        return patch, correctorsList, csi.Kmsij, Rmsij, correctorRhs

    def UpdateElements(self, tol, offset= [], Printing = False):
        print('apply tolerance') if Printing else 1
        Elements_to_be_updated = []
        for (i,eps) in self.E_vh.items():
            if eps > tol:
                if i not in offset:
                    offset.append(i)
                    Elements_to_be_updated.append(i)
        print('... to be updated: {}%'.format(100*np.size(Elements_to_be_updated)/len(self.E_vh)), end='') \
            if Printing else 1

        if np.size(Elements_to_be_updated) != 0:
            print('... update correctors') if Printing else 1
            patchT_irrelevant, correctorsListTNew, KmsijTNew, RmsijTNew, correctorsRhsNew = zip(*map(self.UpdateCorrectors,
                                                                                 Elements_to_be_updated))

            print('replace Kmsij and update correctorsListT') if Printing else 1
            RmsijT_list = list(np.copy(self.RmsijT))
            correctorsRhs_list = list(np.copy(self.correctorsRhsT))
            KmsijT_list = list(np.copy(self.KmsijT))
            correctorsListT_list = list(np.copy(self.correctorsListT))
            i = 0
            for T in Elements_to_be_updated:
                KmsijT_list[T] = KmsijTNew[i]
                correctorsListT_list[T] = correctorsListTNew[i]
                RmsijT_list[T] = RmsijTNew[i]
                correctorsRhs_list[T] = correctorsRhsNew[i]
                i += 1

            self.KmsijT = tuple(KmsijT_list)
            self.correctorsListT = tuple(correctorsListT_list)
            self.RmsijT = tuple(RmsijT_list)
            self.correctorsRhsT = tuple(correctorsRhs_list)


            return offset, 1 # computed
        else:
            print('... there is nothing to be updated') if Printing else 1
            return offset, 0 # not computed

    def StartAlgorithm(self):
        assert(self.init)    # only start the algorithm once

        # in case not every element is affected, the percentage would be missleading.
        eps_size = np.size(self.E_vh)
        self.E_vh = {i: self.E_vh[i] for i in range(np.size(self.E_vh)) if self.E_vh[i] > 0}
        full_percentage = len(self.E_vh) / eps_size

        world = self.world
        print('starting algorithm ...... ')

        TOL = self.StartingTolerance

        TOLt = []
        to_be_updatedT = []
        energy_errorT = []
        rel_energy_errorT = []
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
                    to_be_updated = np.size(offset) / len(self.E_vh) * 100
                    to_be_updatedT.append(to_be_updated * full_percentage)

                    energy_errorT.append(energy_error)
                    tmp_errorT.append(old_tmp_energy_error)
                    if np.size(offset) / len(self.E_vh) == 1:
                        print('     every corrector has been updated')
                        continue_computing = 0
                        continue
                    else:
                        print('     skipping TOL {}'.format(TOL))
                        TOL *= 3/ 4.
                        continue

            to_be_updated = np.size(offset) / len(self.E_vh) * 100
            to_be_updatedT.append(to_be_updated * full_percentage)

            KFull = pglod.assembleMsStiffnessMatrix(world, self.patchT, self.KmsijT)
            RFull = pglod.assemblePatchFunction(world, self.patchT, self.RmsijT)
            Rf = pglod.assemblePatchFunction(world, self.patchT, self.correctorsRhsT)

            basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)

            bFull = basis.T * self.MFull * self.f_trans - RFull

            basisCorrectors = pglod.assembleBasisCorrectors(world, self.patchT, self.correctorsListT)
            modifiedBasis = basis - basisCorrectors

            uFull, _ = pglod.solve(world, KFull, bFull, self.boundaryConditions)

            uLodFine = modifiedBasis * uFull
            uLodFine += Rf

            uFineFull_trans_LOD = uLodFine

            if self.init:
                uFineFull_trans_LOD_old = uLodFine
                self.init = 0

            energy_norm = np.sqrt(np.dot(uFineFull_trans_LOD, self.AFine_trans * uFineFull_trans_LOD))
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

            print('             TOL: {:f}, updates: {:7.3f}%, energy error: {:f}, tmp_error: {:f}, relative energy error: {:f}'.format(TOL,
                                                                                       to_be_updated * full_percentage,
                                                                                       energy_error,
                                                                                       tmp_energy_error, energy_error/energy_norm))

            rel_energy_errorT.append(energy_error/energy_norm)
            energy_errorT.append(energy_error)
            tmp_errorT.append(tmp_energy_error)

            if tmp_energy_error > 1e-5:
                TOL *= 3 / 4.
            else:
                if int(np.size(offset) / len(self.E_vh)) == 1:
                    if computed:
                        print('     stop computing')
                        continue_computing = 0

        return to_be_updatedT, energy_errorT, tmp_errorT, rel_energy_errorT, TOLt, uFineFull_trans_LOD


class PercentageVsErrorAlgorithm:
    def __init__(self, world, k, boundaryConditions, a_Fine_to_be_approximated, aFine_ref, f_trans, E_vh, KmsijT,
                 correctorsListT, patchT, RmsijT, correctorsRhsT, MFull, uFineFull_trans, AFine_trans, E_f = [],
                 computing_options='both'):
        self.world = world
        self.k = k
        self.boundaryConditions = boundaryConditions
        self.a_Fine_to_be_approximated = a_Fine_to_be_approximated
        self.aFine_ref = aFine_ref
        self.f_trans = f_trans
        self.E_vh = E_vh
        self.E_f = E_f
        self.KmsijT = KmsijT
        self.correctorsListT = correctorsListT
        self.patchT = patchT
        self.RmsijT = RmsijT
        self.correctorsRhsT = correctorsRhsT
        self.MFull = MFull
        self.uFineFull_trans = uFineFull_trans
        self.AFine_trans = AFine_trans
        self.computing_options = computing_options

        self.init = 1

    def UpdateCorrectors(self, TInd):
        # print(" UPDATING {}".format(TInd))
        patch = Patch(self.world, self.k, TInd)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, self.boundaryConditions)
        rPatch = lambda: coef.localizeCoefficient(patch, self.a_Fine_to_be_approximated)

        MRhsList = [self.f_trans[util.extractElementFine(self.world.NWorldCoarse,
                                                  self.world.NCoarseElement,
                                                  patch.iElementWorldCoarse,
                                                  extractElements=False)]];

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, rPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, rPatch)

        correctorRhs = lod.computeElementCorrector(patch, IPatch, rPatch, None, MRhsList)[0]
        Rmsij = lod.computeRhsCoarseQuantities(patch, correctorRhs, rPatch)

        return patch, correctorsList, csi.Kmsij, Rmsij, correctorRhs

    def UpdateNextElement(self, tol, offset= [], Printing = False):
        print('apply tolerance') if Printing else 1
        Elements_to_be_updated = []
        for (i,eps) in self.E_vh.items():
            if eps > tol:
                if i not in offset:
                    offset.append(i)
                    Elements_to_be_updated.append(i)
                    break
        print('... to be updated: {}%'.format(100*np.size(Elements_to_be_updated)/len(self.E_vh)), end='') \
            if Printing else 1

        if np.size(Elements_to_be_updated) != 0:
            # assert(np.size(Elements_to_be_updated) == 1 or np.size(Elements_to_be_updated) == 2) # sometimes we get
            print('... update correctors') if Printing else 1
            patchT_irrelevant, correctorsListTNew, KmsijTNew, RmsijTNew, correctorsRhsNew = zip(*map(self.UpdateCorrectors,
                                                                                 Elements_to_be_updated))

            print('replace Kmsij and update correctorsListT') if Printing else 1
            RmsijT_list = list(np.copy(self.RmsijT))
            correctorsRhs_list = list(np.copy(self.correctorsRhsT))
            KmsijT_list = list(np.copy(self.KmsijT))
            correctorsListT_list = list(np.copy(self.correctorsListT))
            i = 0
            for T in Elements_to_be_updated:
                KmsijT_list[T] = KmsijTNew[i]
                correctorsListT_list[T] = correctorsListTNew[i]
                RmsijT_list[T] = RmsijTNew[i]
                correctorsRhs_list[T] = correctorsRhsNew[i]
                i += 1

            self.KmsijT = tuple(KmsijT_list)
            self.correctorsListT = tuple(correctorsListT_list)
            self.RmsijT = tuple(RmsijT_list)
            self.correctorsRhsT = tuple(correctorsRhs_list)

            return offset
        else:
            print('... there is nothing to be updated') if Printing else 1
            return offset

    def StartAlgorithm(self):
        assert(self.init)    # only start the algorithm once

        # in case not every element is affected, the percentage would be missleading.
        eps_size = np.size(self.E_vh)
        self.E_vh = {i: self.E_vh[i] for i in range(np.size(self.E_vh)) if self.E_vh[i] > 0}
        list = [ v for v in self.E_vh.values()]
        list.append(0)
        tols = np.sort(list)[::-1]

        eps_size_f = np.size(self.E_vh)
        self.E_f = {i: self.E_f[i] for i in range(np.size(self.E_f)) if self.E_f[i] > 0}
        list_f = [ v for v in self.E_f.values()]
        list_f.append(0)
        tols_f = np.sort(list)[::-1]

        # make sure we only update one element all the time
        for i in range(1,np.size(tols)):
            if tols[i] == tols[i-1]:
                tols[i] -= 1e-7

        for i in range(1, np.size(tols_f)):
            if tols_f[i] == tols_f[i - 1]:
                tols_f[i] -= 1e-7

        full_percentage = len(self.E_vh) / eps_size
        full_percentage_f = len(self.E_f) / eps_size_f

        world = self.world
        print('starting algorithm ...... ')

        TOLt = []
        to_be_updatedT = []
        energy_errorT = []
        rel_energy_errorT = []
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
                to_be_updated = np.size(offset) / len(self.E_vh) * 100
                to_be_updatedT.append(to_be_updated)
                pass
            else:
                to_be_updated = np.size(offset) / len(self.E_vh) * 100
                to_be_updatedT.append(to_be_updated * full_percentage)

            KFull = pglod.assembleMsStiffnessMatrix(world, self.patchT, self.KmsijT)
            RFull = pglod.assemblePatchFunction(world, self.patchT, self.RmsijT)
            Rf = pglod.assemblePatchFunction(world, self.patchT, self.correctorsRhsT)

            basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)

            bFull = basis.T * self.MFull * self.f_trans - RFull

            basisCorrectors = pglod.assembleBasisCorrectors(world, self.patchT, self.correctorsListT)
            modifiedBasis = basis - basisCorrectors

            uFull, _ = pglod.solve(world, KFull, bFull, self.boundaryConditions)

            uLodFine = modifiedBasis * uFull
            uLodFine += Rf

            uFineFull_trans_LOD = uLodFine

            if self.init:
                self.init = 0
                uFineFull_trans_LOD_old = uLodFine

            energy_norm = np.sqrt(np.dot(uFineFull_trans_LOD, self.AFine_trans * uFineFull_trans_LOD))
            # tmp_error
            tmp_energy_error = np.sqrt(
                np.dot((uFineFull_trans_LOD - uFineFull_trans_LOD_old),
                       self.AFine_trans * (uFineFull_trans_LOD - uFineFull_trans_LOD_old)))


            # actual error
            energy_error = np.sqrt(
                np.dot((uFineFull_trans_LOD - self.uFineFull_trans),
                       self.AFine_trans * (uFineFull_trans_LOD - self.uFineFull_trans)))

            uFineFull_trans_LOD_old = uFineFull_trans_LOD

            print(' step({:3d}/{})    TOL: {:f}, updates: {:7.3f}%, energy error: {:f}, tmp_error: {:f}, relative energy error: {:f}'.format(i, np.size(tols), TOL,
                                                                                       to_be_updated * full_percentage,
                                                                                       energy_error,
                                                                                       tmp_energy_error, energy_error/energy_norm))

            rel_energy_errorT.append(energy_error/energy_norm)
            energy_errorT.append(energy_error)
            tmp_errorT.append(tmp_energy_error)

            if TOL == 0:
                # stop now
                break

        return to_be_updatedT, energy_errorT, tmp_errorT,rel_energy_errorT, TOLt, uFineFull_trans_LOD


class PercentageVsErrorAlgorithm_NO_TOLS:
    def __init__(self, world, k, boundaryConditions, a_Fine_to_be_approximated, aFine_ref, f_trans, E_vh, KmsijT,
                 correctorsListT, patchT, RmsijT, correctorsRhsT, MFull, uFineFull_trans, AFine_trans):
        self.world = world
        self.k = k
        self.boundaryConditions = boundaryConditions
        self.a_Fine_to_be_approximated = a_Fine_to_be_approximated
        self.aFine_ref = aFine_ref
        self.f_trans = f_trans
        self.E_vh = E_vh
        self.KmsijT = KmsijT
        self.correctorsListT = correctorsListT
        self.patchT = patchT
        self.RmsijT = RmsijT
        self.correctorsRhsT = correctorsRhsT
        self.MFull = MFull
        self.uFineFull_trans = uFineFull_trans
        self.AFine_trans = AFine_trans

        self.init = 1

    def UpdateCorrectors(self, TInd):
        # print(" UPDATING {}".format(TInd))
        patch = Patch(self.world, self.k, TInd)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, self.boundaryConditions)
        rPatch = lambda: coef.localizeCoefficient(patch, self.a_Fine_to_be_approximated)

        MRhsList = [self.f_trans[util.extractElementFine(self.world.NWorldCoarse,
                                                  self.world.NCoarseElement,
                                                  patch.iElementWorldCoarse,
                                                  extractElements=False)]];

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, rPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, rPatch)

        correctorRhs = lod.computeElementCorrector(patch, IPatch, rPatch, None, MRhsList)[0]
        Rmsij = lod.computeRhsCoarseQuantities(patch, correctorRhs, rPatch)

        return patch, correctorsList, csi.Kmsij, Rmsij, correctorRhs

    def UpdateNextElement(self, offset= [], Printing = False):
        print('apply tolerance') if Printing else 1
        Element_to_be_updated = []
        for (i,eps) in self.E_vh.items():
            if i not in offset:
                offset.append(i)
                Element_to_be_updated.append(i)
                break
        print('... to be updated: {}%'.format(100*np.size(offset)/len(self.E_vh)), end='') \
            if Printing else 1

        if np.size(Element_to_be_updated) != 0:
            # assert(np.size(Elements_to_be_updated) == 1 or np.size(Elements_to_be_updated) == 2) # sometimes we get
            print('... update correctors') if Printing else 1
            patchT_irrelevant, correctorsListTNew, KmsijTNew, RmsijTNew, correctorsRhsNew = zip(*map(self.UpdateCorrectors,
                                                                                 Element_to_be_updated))

            print('replace Kmsij and update correctorsListT') if Printing else 1
            RmsijT_list = list(np.copy(self.RmsijT))
            correctorsRhs_list = list(np.copy(self.correctorsRhsT))
            KmsijT_list = list(np.copy(self.KmsijT))
            correctorsListT_list = list(np.copy(self.correctorsListT))
            i = 0
            for T in Element_to_be_updated:
                KmsijT_list[T] = KmsijTNew[i]
                correctorsListT_list[T] = correctorsListTNew[i]
                RmsijT_list[T] = RmsijTNew[i]
                correctorsRhs_list[T] = correctorsRhsNew[i]
                i += 1

            self.KmsijT = tuple(KmsijT_list)
            self.correctorsListT = tuple(correctorsListT_list)
            self.RmsijT = tuple(RmsijT_list)
            self.correctorsRhsT = tuple(correctorsRhs_list)

            return offset
        else:
            print('... there is nothing to be updated') if Printing else 1
            return offset

    def StartAlgorithm(self):
        assert(self.init)    # only start the algorithm once

        # in case not every element is affected, the percentage would be missleading.
        eps_size = np.size(self.E_vh)
        self.E_vh = {i: self.E_vh[i] for i in range(np.size(self.E_vh)) if self.E_vh[i] > 0}

        full_percentage = len(self.E_vh) / eps_size

        world = self.world
        print('starting algorithm ...... ')

        TOLt = []
        to_be_updatedT = []
        energy_errorT = []
        rel_energy_errorT = []
        tmp_errorT = []

        offset = []
        TOL = 100   # not relevant

        for i in range(len(self.E_vh)+1):
            if self.init:
                pass
            else:
                offset = self.UpdateNextElement(offset, Printing=False)

            if self.init:
                to_be_updated = np.size(offset) / len(self.E_vh) * 100
                to_be_updatedT.append(to_be_updated)
                pass
            else:
                to_be_updated = np.size(offset) / len(self.E_vh) * 100
                to_be_updatedT.append(to_be_updated * full_percentage)

            KFull = pglod.assembleMsStiffnessMatrix(world, self.patchT, self.KmsijT)
            RFull = pglod.assemblePatchFunction(world, self.patchT, self.RmsijT)
            Rf = pglod.assemblePatchFunction(world, self.patchT, self.correctorsRhsT)

            basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)

            bFull = basis.T * self.MFull * self.f_trans - RFull

            basisCorrectors = pglod.assembleBasisCorrectors(world, self.patchT, self.correctorsListT)
            modifiedBasis = basis - basisCorrectors

            uFull, _ = pglod.solve(world, KFull, bFull, self.boundaryConditions)

            uLodFine = modifiedBasis * uFull
            uLodFine += Rf

            uFineFull_trans_LOD = uLodFine

            if self.init:
                uFineFull_trans_LOD_old = uLodFine

            energy_norm = np.sqrt(np.dot(uFineFull_trans_LOD, self.AFine_trans * uFineFull_trans_LOD))
            # tmp_error
            tmp_energy_error = np.sqrt(
                np.dot((uFineFull_trans_LOD - uFineFull_trans_LOD_old),
                       self.AFine_trans * (uFineFull_trans_LOD - uFineFull_trans_LOD_old)))


            # actual error
            energy_error = np.sqrt(
                np.dot((uFineFull_trans_LOD - self.uFineFull_trans),
                       self.AFine_trans * (uFineFull_trans_LOD - self.uFineFull_trans)))

            uFineFull_trans_LOD_old = uFineFull_trans_LOD

            if self.init:
                self.init = 0
                print(
                    ' step({:3d}/{})  T: {}  updates: {:7.3f}%, energy error: {:f}, tmp_error: {:f}, relative energy error: {:f}'.format(
                        i, len(self.E_vh), ' - ',
                        to_be_updated * full_percentage,
                        energy_error,
                        tmp_energy_error, energy_error / energy_norm))
            else:
                print(' step({:3d}/{})  T: {:3d}  updates: {:7.3f}%, energy error: {:f}, tmp_error: {:f}, relative energy error: {:f}'.format(i, len(self.E_vh), offset[-1],
                                                                                       to_be_updated * full_percentage,
                                                                                       energy_error,
                                                                                       tmp_energy_error, energy_error/energy_norm))

            rel_energy_errorT.append(energy_error/energy_norm)
            energy_errorT.append(energy_error)
            tmp_errorT.append(tmp_energy_error)


        return to_be_updatedT, energy_errorT, tmp_errorT,rel_energy_errorT, TOLt, uFineFull_trans_LOD