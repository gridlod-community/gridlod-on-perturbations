# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

class smooth_1d:
    # the perturbation is defined as
    # y = psi(x) = x + alpha * (1-x)x
    def __init__(self, NFine, alpha):
        self.NFine = NFine
        self.alpha = alpha

    def inverse_evaluate(self, x):
        alpha = self.alpha
        return (alpha+1)/(2 *alpha) - np.sqrt(alpha**2-4*alpha*x+2*alpha+1)/(2*alpha)

    def evaluate(self, x):
        alpha = self.alpha
        return x + alpha * (1-x)*x

    def J(self, x):
        alpha = self.alpha
        return (-2*alpha*x) + alpha + 1

    def detJ(self, x):
        return self.J(x)

    def transformation(self, function, x_values):
        assert(np.size(function) == np.size(x_values))
        x_transformed = self.evaluate(x_values)
        new_function = np.copy(function)
        for i in range(np.shape(x_values)[0]):
            new_function[i] = function[int(x_transformed[i] * self.NFine)]
        return new_function

    def inverse_transformation(self, function, x_values):
        assert(np.size(function) == np.size(x_values))
        x_transformed = self.inverse_evaluate(x_values)
        new_function = np.copy(function)
        for i in range(np.shape(x_values)[0]):
            new_function[i] = function[int(x_transformed[i] * self.NFine)]
        return new_function

    def apply_transformation_to_bilinear_form(self, aFine_in, x_values):
        assert (np.size(aFine_in) == np.size(x_values))
        aFine_out = np.copy(aFine_in)
        J = self.J(x_values)
        detJ = self.detJ(x_values)
        for i in range(np.shape(x_values)[0]):
            aFine_out[i] *= detJ[i] / (J[i] * J[i])  # NOTE: one dimensional case
        return aFine_out

    def apply_transformation_to_linear_functional(self, f_in, x_values):
        assert (np.size(f_in) == np.size(x_values))
        f_out = np.copy(f_in)
        detJ = self.detJ(x_values)
        for i in range(np.shape(x_values)[0]):
            f_out[i] *= detJ[i]
        return f_out


class plateau_1d:
    # the perturbation is a plateau with
    # y = psi(x) = x + alpha * xi(x)
    # xi(x) =           ______      1/2
    #                 /        \
    #               /            \
    #              1 1/4 1/2 3/4  1

    def __init__(self, alpha):
        self.alpha = alpha

    def inverse_evaluate(self, x):
        alpha = self.alpha
        if (x < 1. / 4. + alpha / 2):
            return x * (1 / (1 + 2 * alpha))
        if ((1. / 4. + alpha / 2 <= x) & (x <= 3 / 4. + alpha / 2)):
            return x - alpha / 2.
        if (x > 3 / 4. + alpha / 2):
            return (x - alpha * 2.) / (1 - 2. * alpha)

    def evaluate(self, x):
        alpha = self.alpha
        if (x < 1. / 4.):
            return x + alpha*2*x
        if ((1. / 4. <= x) & (x <= 3 / 4. )):
            return x + alpha/2
        if (x > 3 / 4.):
            return x + alpha * (2. - 2. * x)

    def J(self, x):
        alpha = self.alpha
        if (x < 1. / 4.):
            return (1. + 2 * alpha)
        if ((1. / 4. <= x) & (x <= 3 / 4.)):
            return 1.
        if (x > 3 / 4.):
            return (1. - 2 * alpha)

    def detJ(self, x):
        return self.J(x)

class plateau_2d_on_one_axis:
    # the perturbation is a plateau with
    # y_1, y_2 = psi(x_1,x_2) = (x_1,x_2) + alpha * xi(x_1,x_2)
    # xi_1(x) =         ______      1/2
    #                 /        \
    #               /            \
    #              1 1/4 1/2 3/4  1
    #
    # and xi_2(x) = 0

    def __init__(self, alpha):
        self.alpha = alpha

    def inverse_evaluate(self, x):
        assert(np.size(x)==2)
        alpha = self.alpha
        ret = np.array([0.,0.])
        if (x[0] < 1. / 4. + alpha / 2):
            ret[0] = x[0] * (1 / (1 + 2 * alpha))
        if ((1. / 4. + alpha / 2 <= x[0]) & (x[0] <= 3 / 4. + alpha / 2)):
            ret[0] = x[0] - alpha / 2.
        if (x[0] > 3 / 4. + alpha / 2):
            ret[0] = (x[0] - alpha * 2.) / (1 - 2. * alpha)
        ret[1] = x[1]
        return ret

    def evaluate(self, x):
        assert (np.size(x) == 2)
        alpha = self.alpha
        ret = np.array([0., 0.])
        if (x[0] < 1. / 4.):
            ret[0] = x[0] + alpha*2*x[0]
        if ((1. / 4. <= x[0]) & (x[0] <= 3 / 4. )):
            ret[0] = x[0] + alpha/2
        if (x[0] > 3 / 4.):
            ret[0] = x[0] + alpha * (2. - 2. * x[0])
        ret[1] = x[1]
        return ret

    def J(self, x):
        assert(np.size(x)==2)
        alpha = self.alpha
        ret = np.array([[0.,0.],[0.,0.]])
        if (x[0] < 1. / 4.):
            ret[0,0] = (1. + 2 * alpha)
        if ((1. / 4. <= x[0]) & (x[0] <= 3 / 4.)):
            ret[0,0] = 1.
        if (x[0] > 3 / 4.):
            ret[0,0] = (1. - 2 * alpha)
        ret[1,1] = 1
        return ret

    def Jinv(self, x):
        J = self.J(x)
        return np.linalg.inv(J)

    def detJ(self, x):
        J = self.J(x)
        return np.linalg.det(J)