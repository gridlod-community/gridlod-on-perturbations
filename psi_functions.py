# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

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