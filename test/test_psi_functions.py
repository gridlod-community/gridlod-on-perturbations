import unittest
import numpy as np

import psi_functions

def util_inverse_correct(testCase, psi):
    np.random.seed(0)
    
    # Test batch of points
    x_points = np.random.rand(100, psi.d)
    testCase.assertTrue(np.allclose(psi.inverse_evaluate(psi.evaluate(x_points)), x_points))

    # Test single point
    x_points = np.random.rand(psi.d)
    testCase.assertTrue(np.allclose(psi.inverse_evaluate(psi.evaluate(x_points)), x_points))

def util_J_correct(testCase, psi):
    np.random.seed(0)
    x_points = np.random.rand(100, psi.d)
    directions = np.random.rand(100, psi.d)
    directions = directions/np.linalg.norm(directions, axis=1)[:,None]
    h = 1e-7

    psi_of_xs = psi.evaluate(x_points)
    psi_of_xhs = psi.evaluate(x_points + h*directions)
    approx_directional_derivatives = (psi_of_xhs - psi_of_xs)/h
    J_directional_derivatives = np.einsum('ijk,ik->ij', psi.J(x_points), directions)

    testCase.assertTrue(np.allclose(approx_directional_derivatives, J_directional_derivatives))

class plateau_2d_on_one_axis_TestCase(unittest.TestCase):

    def test_inverse_correct(self):
        alpha = 1./4
        NFine = None
        plateau = psi_functions.plateau_2d_on_one_axis(NFine, alpha)
        util_inverse_correct(self, plateau)
        
    def test_J_correct(self):
        alpha = 1./4
        NFine = None
        plateau = psi_functions.plateau_2d_on_one_axis(NFine, alpha)
        util_J_correct(self, plateau)
        
if __name__ == '__main__':
    unittest.main()
