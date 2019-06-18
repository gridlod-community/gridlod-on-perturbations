import numpy as np
import random
import matplotlib.pyplot as plt

from gridlod import util, func

from gridlod_on_perturbations import discrete_mapping

class PerturbationInterface:
    def __init__(self, world):
        self.world = world

    def computePerturbation(self, aFine, f_ref):
        assert(self.psi)
        assert(len(aFine) == np.prod(self.world.NWorldFine))

        NFine = self.world.NWorldFine

        # Elements
        xtFine = util.tCoordinates(NFine)
        xtFine_ref = self.psi.inverse_evaluate(xtFine)
        xtFine_pert = self.psi.evaluate(xtFine_ref)

        # Nodes
        xpFine = util.pCoordinates(NFine)
        xpFine_ref = self.psi.inverse_evaluate(xpFine)
        xpFine_pert = self.psi.evaluate(xpFine_ref)

        self.checkInvertible(xpFine_pert, xpFine)

        aFine_pert = func.evaluateDQ0(NFine, aFine, xtFine_ref)
        f_pert = func.evaluateCQ1(NFine, f_ref, xpFine_ref)
        return aFine_pert, f_pert

    def checkInvertible(self, aFine, aBack):
        if np.allclose(aBack, aFine):
            print('psi is invertible')
        else:
            norm = np.linalg.norm(aBack - aFine)
            if norm <= 1e-4:
                print('psi is almost invertible, the norm is {}'.format(norm))
            else:
                print('psi is not invertible, the norm is {} !'.format(norm))

    def computeTransformation(self, aFine, f_ref):
        assert(len(aFine) == np.prod(self.world.NWorldFine))
        assert(len(f_ref) == np.prod(self.world.NWorldFine+1))
        assert(self.psi)
        psi = self.psi

        NFine = self.world.NWorldFine
        xtFine = util.tCoordinates(NFine)
        xpFine = util.pCoordinates(NFine)

        a_trans = np.einsum('tij, t, tkj, t -> tik', psi.Jinv(xtFine), aFine, psi.Jinv(xtFine),
                                psi.detJ(xtFine))
        f_trans = np.einsum('t, t -> t', f_ref, psi.detJ(xpFine))

        return a_trans, f_trans

    def evaluateSolution(self, u):
        NFine = self.world.NWorldFine

        xpFine = util.pCoordinates(NFine)
        xpFine_ref = self.psi.inverse_evaluate(xpFine)

        return func.evaluateCQ1(NFine, u, xpFine_ref)

class BendingInOneArea(PerturbationInterface):
    # TODO: Generalize concept for arbitrary regions. Base Class for BendingInTwoAreas and so on.
    def __init__(self, world, area=[0,0.25], bending_factor=1):
        self.area = area
        self.bending_factor = bending_factor
        super().__init__(world)
        self.create()

    def create(self):
        NFine = self.world.NWorldFine
        fine = NFine[0]

        xpFine = util.pCoordinates(NFine)

        forward_mapping = np.stack([xpFine[:, 0], xpFine[:, 1]], axis=1)

        xpFine_shaped = xpFine.reshape(fine + 1, fine + 1, 2)

        left = int((fine+1) * self.area[0])
        right = int((fine+1) * self.area[1])

        # TODO: enable other left_2 and right_2
        left_2, right_2 = left, right

        print('left_right ', left, right)

        part_x = xpFine_shaped[left:right, left_2:right_2, 0]
        part_y = xpFine_shaped[left:right, left_2:right_2, 1]
        left_margin_x = np.min(part_x)
        right_margin_x = np.max(part_x)
        left_margin_y = np.min(part_y)
        right_margin_y = np.max(part_y)

        print(left_margin_x, right_margin_x, left_margin_y, right_margin_y)

        epsilon = self.bending_factor / (right_margin_y - left_margin_y)  # why does this have to be so large???

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

        self.psi = discrete_mapping.MappingCQ1(NFine, forward_mapping)

class BendingInTwoAreas(PerturbationInterface):
    # TODO: Generalize concept for arbitrary regions. BaseClass for BendingInOneArea BendingInTwoAreas and so on.
    def __init__(self, world, bending_factor=20):
        self.bending_factor = bending_factor
        super().__init__(world)
        self.create()

    def create(self):
        NFine = self.world.NWorldFine
        fine = NFine[0]
        xpFine = util.pCoordinates(NFine)

        forward_mapping = np.stack([xpFine[:, 0], xpFine[:, 1]], axis=1)

        xpFine_shaped = xpFine.reshape(fine + 1, fine + 1, 2)

        for i in [1, 3]:
            middle = int((fine + 1) * (i / 4))
            intervall = int((fine + 1) / 8)

            left_2 = middle - int(intervall)
            right_2 = middle + int(intervall)

            left, right = left_2, right_2

            # print(fine + 1, left, right)

            part_x = xpFine_shaped[left:right, left_2:right_2, 0]
            part_y = xpFine_shaped[left:right, left_2:right_2, 1]
            left_margin_x = np.min(part_x)
            right_margin_x = np.max(part_x)
            left_margin_y = np.min(part_y)
            right_margin_y = np.max(part_y)

            # print(left_margin_x, right_margin_x, left_margin_y, right_margin_y)

            epsilon = self.bending_factor / (right_margin_y - left_margin_y)  # why does this have to be so large???

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

        self.psi = discrete_mapping.MappingCQ1(NFine, forward_mapping)

class Oscillation(PerturbationInterface):
    def __init__(self, world, num_dots):
        super().__init__(world)
        self.num_dots = num_dots
        self.create()

    def create(self):
        NFine = self.world.NWorldFine
        fine = NFine[0]
        x = util.pCoordinates(NFine)

        num_dots = self.num_dots
        epsilon = 0.4
        frequency = 2*np.pi*num_dots
        
        displacement_unscaled = 1./(frequency)*np.column_stack([0.5*(np.sin(0.5*frequency*x[:,0])+1)*np.sin(frequency*x[:,0]),
                                                                0.5*(np.sin(0.5*frequency*x[:,1])+1)*np.sin(frequency*x[:,1])])
        point_tile_index = np.array(np.minimum(np.floor(num_dots*x), num_dots-1), dtype=int)
        tile_scaling = epsilon*2*np.random.rand(num_dots, num_dots)**2
        point_scaling = tile_scaling[point_tile_index[:,0], point_tile_index[:,1]]
        
        displacement = displacement_unscaled*point_scaling[...,None]
        
        self.psi = discrete_mapping.MappingCQ1(NFine, x + displacement)

class Pinch(PerturbationInterface):
    def __init__(self, world):
        super().__init__(world)
        self.create()

    def create(self):
        NFine = self.world.NWorldFine
        fine = NFine[0]
        x = util.pCoordinates(NFine)

        x0 = np.array([0.5, 0.5])

        r = np.linalg.norm((x-x0), axis=1)
        r_min = 0.1
        r_bounded = np.maximum(r, r_min)/r_min

        tapering = np.prod(np.sin(np.pi*x), axis=1)
        
        displacement = np.column_stack([tapering*0.02*(r_bounded)**-2,
                                        tapering*0.02*(r_bounded)**-2])
        
        self.psi = discrete_mapping.MappingCQ1(NFine, x + displacement)
        
class MultipleBendingStripes(PerturbationInterface):
    def __init__(self, world, number_of_channels, ref_array, space, thick):
        # TODO: Enhace input arguments
        self.number_of_channels = number_of_channels
        self.ref_array = ref_array
        self.space = space
        self.thick = thick
        super().__init__(world)
        self.create()

    def create(self):
        NFine = self.world.NWorldFine
        fine = NFine[0]
        xpFine = util.pCoordinates(NFine)

        epsilonT = []

        forward_mapping = np.stack([xpFine[:, 0], xpFine[:, 1]], axis=1)

        xpFine_shaped = xpFine.reshape(fine + 1, fine + 1, 2)
        left, right = 0, fine + 1

        for c in range(self.number_of_channels):
            count = 0
            for i in range(np.size(self.ref_array)):
                if self.ref_array[i] == 1:
                    count += 1
                if count == (c + 1) * self.thick:
                    begin = i + 1 - self.space // 2
                    end = i + 1 + self.thick + self.space // 2
                    break
            print(begin, end)
            left_2, right_2 = begin, end
            if c == 3:
                epsilon = 25
            # elif c == 4:
            #    epsilon = -25
            # elif c % 2 == 0:
            #    epsilon = 0
            else:
                epsilon = np.random.uniform(-10, 10)

            part_x = xpFine_shaped[left:right, left_2:right_2, 0]
            part_y = xpFine_shaped[left:right, left_2:right_2, 1]
            left_margin_x = np.min(part_x)
            right_margin_x = np.max(part_x)
            left_margin_y = np.min(part_y)
            right_margin_y = np.max(part_y)

            print(left_margin_x, right_margin_x, left_margin_y, right_margin_y)

            forward_mapping_partial = np.stack([xpFine_shaped[left:right, left_2:right_2, 0]
                                                + epsilon *
                                                (xpFine_shaped[left:right, left_2:right_2, 0] - left_margin_x) *
                                                (right_margin_x - xpFine_shaped[left:right, left_2:right_2, 0]) *
                                                (xpFine_shaped[left:right, left_2:right_2, 1] - left_margin_y) *
                                                (right_margin_y - xpFine_shaped[left:right, left_2:right_2, 1]),
                                                xpFine_shaped[left:right, left_2:right_2, 1]], axis=2)

            forward_mapping_shaped = forward_mapping.reshape(fine + 1, fine + 1, 2)
            forward_mapping_shaped[left:right, left_2:right_2, :] = forward_mapping_partial

            epsilonT.append(epsilon)

        forward_mapping = forward_mapping_shaped.reshape((fine + 1) ** 2, 2)

        print('Those are the results of the shift epsilon', epsilonT)

        self.psi = discrete_mapping.MappingCQ1(NFine, forward_mapping)

class MultipleMovingStripes(PerturbationInterface):
    def __init__(self, world, number_of_channels, space, thick, plot_mapping = False):
        # TODO: Enhace input arguments !!
        self.number_of_channels = number_of_channels
        self.space = space
        self.thick = thick
        self.plot_mapping = plot_mapping
        super().__init__(world)
        self.create()

    def create(self):
        NFine = self.world.NWorldFine
        fine = NFine[0]
        xpFine = util.pCoordinates(NFine)

        Nmapping = np.array([int(fine), int(fine)])

        size_of_an_element = 1. / fine
        print('the size of a fine element is {}'.format(size_of_an_element))
        walk_with_perturbation = size_of_an_element

        epsilonT = []
        cq1 = np.zeros((int(fine) + 1, int(fine) + 1))

        cs = np.random.randint(0, 2, self.number_of_channels)
        cs = [c * random.sample([-1, 1], 1)[0] for c in cs]

        # or manually
        cs[3] = 10
        cs[4] = 2
        cs[5] = 1

        print(cs)

        last_value = 0
        for i, c in enumerate(cs):
            platform = self.space // 2 + 2 * self.thick
            begin = platform // 2 + i * (self.space + self.thick)
            end = begin + self.space - platform + self.thick

            epsilon = c * walk_with_perturbation
            epsilonT.append(epsilon)
            walk = epsilon - last_value

            constant_length = platform + self.thick
            increasing_length = end - begin

            for i in range(increasing_length):
                cq1[:, begin + i] = last_value + (i + 1) / increasing_length * walk

            for i in range(constant_length):
                cq1[:, begin + increasing_length + i] = epsilon

            last_value = epsilon

        # ending
        begin += self.space + self.thick
        end = begin + self.space - platform + self.thick
        epsilon = 0
        walk = epsilon - last_value
        increasing_length = end - begin
        for i in range(increasing_length):
            cq1[:, begin + i] = last_value + (i + 1) / increasing_length * walk

        if self.plot_mapping:
            plt.plot(np.arange(0, fine + 1), cq1[self.space, :], label='$id(x) - \psi(x)$')
            plt.title('Domain mapping')
            plt.legend()
            plt.show()

        print('These are the results of the shift epsilon', epsilonT)
        cq1 = cq1.flatten()

        alpha = 1.

        for_mapping = np.stack((xpFine[:, 0] + alpha * func.evaluateCQ1(Nmapping, cq1, xpFine), xpFine[:, 1]), axis=1)
        self.psi = discrete_mapping.MappingCQ1(NFine, for_mapping)

class StripesWithDots(PerturbationInterface):
    def __init__(self, world, number_of_channels, ref_array, space, thick, plot_mapping = False):
        # TODO: Enhace input arguments
        self.number_of_channels = number_of_channels
        self.ref_array = ref_array
        self.space = space
        self.thick = thick
        self.plot_mapping = plot_mapping
        super().__init__(world)
        self.create()


    def create(self):
        NFine = self.world.NWorldFine
        fine = NFine[0]
        xpFine = util.pCoordinates(NFine)

        number_of_perturbed_channels = 4

        now = 0
        count = 0
        for i in range(np.size(self.ref_array)):
            if self.ref_array[i] == 1:
                count += 1
            if count == 8 * self.thick:  # at the 8ths shape (which is the last dot in one line, the cq starts)
                begin = i + 1
                break
        count = 0
        for i in range(np.size(self.ref_array)):
            if self.ref_array[i] == 1:
                count += 1
            if count == 13 * self.thick - 3:  # it ends after the last channel
                end = i
                break

        # Discrete mapping
        Nmapping = np.array([int(fine), int(fine)])
        cq1 = np.zeros((int(fine) + 1, int(fine) + 1))

        # I only want to perturb on the fine mesh.
        size_of_an_element = 1. / fine
        walk_with_perturbation = size_of_an_element

        channels_position_from_zero = self.space
        channels_end_from_zero = channels_position_from_zero + self.thick

        # The next only have the purpose to make the psi invertible.
        increasing_length = (end - begin) // (number_of_perturbed_channels + 1) - self.thick - 2
        constant_length = (end - begin) - increasing_length * 2
        maximum_walk = (increasing_length - 6) * walk_with_perturbation
        walk_with_perturbation = maximum_walk
        for i in range(increasing_length):
            cq1[:, begin + 1 + i] = (i + 1) / increasing_length * walk_with_perturbation
            cq1[:, begin + increasing_length + i + constant_length] = walk_with_perturbation - (
                        i + 1) / increasing_length * walk_with_perturbation
        for i in range(constant_length):
            cq1[:, begin + increasing_length + i] = walk_with_perturbation

        # Check what purtubation I have
        if self.plot_mapping:
            plt.figure('DomainMapping')
            plt.plot(np.arange(0, fine + 1), cq1[self.space, :], label='$id(x) - \psi(x)$')
            plt.title('Domain mapping')
            plt.legend()

        cq1 = cq1.flatten()

        xpFine = util.pCoordinates(NFine)

        alpha = 1.

        for_mapping = np.stack((xpFine[:, 0] + alpha * func.evaluateCQ1(Nmapping, cq1, xpFine), xpFine[:, 1]), axis=1)
        self.psi = discrete_mapping.MappingCQ1(NFine, for_mapping)
