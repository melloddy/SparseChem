import unittest
from ACE_ECE_calculation_oneTarget_forSparseChem import calculateErrors_oneTarget
import numpy as np
from scipy import sparse


y_class_case1=sparse.csr_matrix(np.array([[1], [-1], [1], [1], [-1]]))
y_hat_case1=sparse.csr_matrix(np.array([[0.12], [0.41], [0.61], [0.71], [0.94]]))


y_class_case2=sparse.csr_matrix(np.array([[-1],[1],[-1],[-1],[1],[-1],[1], [1],[1],[1],[-1],[-1]]))
y_hat_case2=sparse.csr_matrix(np.array([[0.01],[0.04],[0.21],[0.46],[0.46],[0.58],[0.61],[0.77],[0.86],[0.87],[0.89],[0.94]]))
y_class_case3=sparse.csr_matrix(np.array([[-1],[1],[-1],[1],[0],[0],[0], [0],[0],[0],[0],[0]]))
y_hat_case3=sparse.csr_matrix(np.array([[0.41],[0.56],[0.67],[0.90],[0],[0],[0],[0],[0],[0],]))                                 
y_class_case4=sparse.csr_matrix(np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]))
y_hat_case4=sparse.csr_matrix(np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]))


class TestcalculateErrors(unittest.TestCase):
    def test_calculateErrors_case1(self):

        test_true = y_class_case1
        test_score=y_hat_case1
        test_bins=10
        result = calculateErrors_oneTarget(test_true, test_score, test_bins)
        self.assertEqual(result[0], (2.91/5))
        self.assertEqual(result[1], (2.91/5))
    
    def test_calculateErrors_case2(self):

        test_true = y_class_case2
        test_score=y_hat_case2
        test_bins=10
        result = calculateErrors_oneTarget(test_true, test_score, test_bins)
        self.assertAlmostEqual(result[0], (4/12))
        self.assertAlmostEqual(result[1], (5.46/12))

    def test_calculateErrors_case3(self):

        test_true = y_class_case3
        test_score=y_hat_case3
        test_bins=10
        result = calculateErrors_oneTarget(test_true, test_score, test_bins)
        self.assertAlmostEqual(result[0], (1.62/4))
        self.assertAlmostEqual(result[1], (0.405))
    
    def test_calculateErrors_case4(self):

        test_true = y_class_case4
        test_score=y_hat_case4
        test_bins=10
        result = calculateErrors_oneTarget(test_true, test_score, test_bins)
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))

if __name__ == '__main__':
    unittest.main()