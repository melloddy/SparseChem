# Copyright (c) 2020 KU Leuven
import sparsechem as sc
import unittest
import torch
import numpy as np

class TestSparse(unittest.TestCase):
    def test_sparsesplit2(self):
        x = torch.sparse_coo_tensor(
                torch.LongTensor([
                    [0,  0, 1,  1,  1, 2, 2, 3, 3,  3],
                    [5, 20, 9, 10, 14, 0, 7, 0, 7, 19]]),
                torch.randn(10),
                size=[5, 25]
            )

        x0, x1 = sc.sparse_split2(x, 15, dim=1)
        self.assertEqual(x0.shape, (x.shape[0], 15))
        self.assertEqual(x1.shape, (x.shape[0], x.shape[1] - 15))

        d0, d1 = torch.split(x.to_dense(), [15, 10], dim=1)
        self.assertTrue(np.allclose(d0, x0.to_dense()))
        self.assertTrue(np.allclose(d1, x1.to_dense()))

if __name__ == '__main__':
    unittest.main()
