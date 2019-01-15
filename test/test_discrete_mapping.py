import unittest
import numpy as np
from gridlod import util

import discrete_mapping

class invert_cq1_mapping_TestCase(unittest.TestCase):
    def test_smooth_1d(self):
        N = np.array([1000])
        x = util.pCoordinates(N)

        e = np.exp(1)
        mapping = (np.exp(x)-1)/(e-1)
        
        inv_mapping = discrete_mapping.invert_cq1_mapping(N, mapping)
        inv_mapping_should_be = np.log((e-1)*x+1)

        self.assertTrue(np.allclose(inv_mapping, inv_mapping_should_be, atol=1e-3))

    def test_smooth_2d(self):
        N = np.array([100, 100])
        x = util.pCoordinates(N)

        e = np.exp(1)
        mapping = (np.exp([x[:,0] + 0.1*x[:,0]*(1-x[:,0])*x[:,1]*(1-x[:,1]),
                           x[:,1] + 0.1*x[:,1]*(1-x[:,1])*x[:,0]*(1-x[:,0])])-1)/(e-1)

        mapping = np.transpose(mapping)
        
        inv_mapping = discrete_mapping.invert_cq1_mapping(N, mapping)
        inv_inv_mapping = discrete_mapping.invert_cq1_mapping(N, inv_mapping)

        self.assertTrue(np.allclose(mapping, inv_inv_mapping, atol=1e-4))

        
        
if __name__ == '__main__':
    unittest.main()
