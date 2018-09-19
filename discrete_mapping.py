# This file is part of the project for "Localization of multiscale problems with random defects":
#   https://github.com/gridlod-community/gridlod-on-perturbations.git
# Copyright holder: Tim Keil, Fredrik Hellman
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from gridlod import util, func

def invert_cq1_mapping(N, mapping):
    """Invert a y = psi(x) CQ1 mapping using the iterative method "A
    simple fixed-point approach to invert a deformation field", Chen
    et al 2008.
    """
    
    coords = util.pCoordinates(N)
    displace = mapping - coords

    inv_displace = np.zeros_like(displace)
    
    max_iter = 100
    iter = 0

    tolerance = 1e-7
    error = 1
    
    while iter < max_iter and error > tolerance:
        inv_displace_previous = inv_displace
        points = np.maximum(0, np.minimum(1.0, coords + inv_displace))
        inv_displace = -func.evaluateCQ1(N, displace, points)
        iter += 1
        error = np.max(np.abs(inv_displace - inv_displace_previous))

    return inv_displace + coords

class MappingCQ1:
    # This is a psi-function defined in terms of a d-valued cq1-function
    def __init__(self, N, mapping):
        self.N = N
        self.mapping = mapping
        self.inv_mapping = invert_cq1_mapping(N, mapping)
        self.d = N.size

    def evaluate(self, x):
        return func.evaluateCQ1(self.N, mapping, x)

    def inverse_evaluate(self, x):
        return func.evaluateCQ1(self.N, inv_mapping, x)

    def J(self, x):
        return func.evaluateCQ1D(self.N, mapping, x)
    
    def Jinv(self, x):
        return np.linalg.inv(self.J(x))

    def detJ(self, x):
        return np.linalg.det(self.J(x))
