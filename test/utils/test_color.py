import unittest
import torch

from pytools.utils.color import (
    autocov, transform_color_statistics, transport_optimal_statistics,
    gaussian_barycenter, multispectral_barycenter)

class TestAutocov(unittest.TestCase):

    def test_autocov(self):
        x = torch.tensor([
            [[1., -1.], 
             [1., -1.]], 
            [[-1., 0.], 
             [-1., 0.]]])
        exp_mean = torch.tensor([0., -0.5])
        exp_cov = torch.tensor([
            [1., -0.5], 
            [-0.5, .25]
        ])
        mean, cov = autocov(x)

        self.assertTrue(torch.allclose(exp_mean, mean))
        self.assertTrue(torch.allclose(exp_cov, cov))

    def test_autocov_batch(self):
        x = torch.tensor([
            [[[1., -1.], 
              [1., -1.]], 
             [[-1., 0.], 
              [-1., 0.]]],
            [[[1., 3.], 
              [-1., 1.]], 
             [[-1., 0.], 
              [2, 0.]]]
        ])
        exp_mean = torch.tensor([
            [0., -0.5],
            [1., .25]
        ])
        exp_cov = torch.tensor([
            [[1., -0.5], 
             [-0.5, .25]],
            [[2., -1.], 
             [-1., 1.1875]]
        ])
        mean, cov = autocov(x)

        self.assertTrue(torch.allclose(exp_mean, mean))
        self.assertTrue(torch.allclose(exp_cov, cov))

class TestTransformColorStats(unittest.TestCase):

    def test_transform(self):
        print("")
        print("Test transform color stats")
        print("==========================")
        x = torch.randn(3, 100, 100)
        y = torch.randn(3, 100, 100)

        transform, transform_inv = transform_color_statistics(x, y)

        mean_x, cov_x = autocov(x)
        mean_y, cov_y = autocov(y)
        mean_xt, cov_xt = autocov(transform(x))
        mean_yt, cov_yt = autocov(transform_inv(y))

        def assertAllClose(x, y, message, atol=1e-5):
            print(x)
            print(y)
            print(torch.dist(x, y))
            self.assertTrue(torch.allclose(x, y, atol=atol), message)

        assertAllClose(mean_y, mean_xt, "Different means for transform")
        assertAllClose(cov_y, cov_xt, "Different covariances for transform")
        assertAllClose(
            mean_x, mean_yt, "Different means for inverse transform")
        assertAllClose(
            cov_x, cov_yt, "Different covariances for inverse transform")

    def test_transform_batch(self):
        print("")
        print("Test transform color stats batch")
        print("================================")
        x = torch.randn(8, 3, 100, 100)
        y = torch.randn(8, 3, 100, 100)

        transform, transform_inv = transform_color_statistics(x, y)

        mean_x, cov_x = autocov(x)
        mean_y, cov_y = autocov(y)
        mean_xt, cov_xt = autocov(transform(x))
        mean_yt, cov_yt = autocov(transform_inv(y))

        self.assertTrue(
            torch.allclose(mean_y, mean_xt, atol=1e-6), 
            f"Different means for transform {torch.max(mean_y - mean_xt)}"
        )
        self.assertTrue(
            torch.allclose(cov_y, cov_xt, atol=1e-6), 
            f"Different covariances for transform {torch.max(cov_y - cov_xt)}"
        )
        self.assertTrue(
            torch.allclose(mean_x, mean_yt, atol=1e-6),
            f"Different means for inverse transform \
{torch.max(mean_x - mean_yt)}"
        )
        self.assertTrue(
            torch.allclose(cov_x, cov_yt, atol=1e-6),
            f"Different covariances for inverse transform \
{torch.max(mean_x - mean_yt)}"
        )

    def test_transport_optimal(self):
        print("")
        print("Test transport optimal")
        print("======================")
        x = torch.randn(3, 100, 100)
        y = torch.randn(3, 100, 100)

        transform, transform_inv = transport_optimal_statistics(x, y)

        mean_x, cov_x = autocov(x)
        mean_y, cov_y = autocov(y)
        mean_xt, cov_xt = autocov(transform(x))
        mean_yt, cov_yt = autocov(transform_inv(y))

        def assertAllClose(x, y, message, atol=1e-5):
            print(x)
            print(y)
            print(torch.dist(x, y))
            self.assertTrue(torch.allclose(x, y, atol=atol), message)

        assertAllClose(mean_y, mean_xt, "Different means for transform")
        assertAllClose(cov_y, cov_xt, "Different covariances for transform")
        assertAllClose(
            mean_x, mean_yt, "Different means for inverse transform")
        assertAllClose(
            cov_x, cov_yt, "Different covariances for inverse transform")
        
class TestBarycenter(unittest.TestCase):

    def test_multispectral_barycenter(self):
        print("")
        print("Test multispectral_barycenter")
        print("=============================")
        img = torch.load("/scratchm/sollivie/data/sentinel2/gatys_512x512/S2A_MSIL2A_20230615T102031_N0509_R065_T30PZR_20230615T200553.SAFE.pt")
        _, sigma = autocov(img)
        barycenter = multispectral_barycenter(sigma)

if __name__ == "__main__":
    unittest.main()