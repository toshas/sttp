import unittest

import torch

from src.utils.spectral_tensors_factory import SpectralTensorsFactorySTTP, SpectralTensorsFactorySVDP, \
    SpectralConvNd
from src.utils.stiefel_parameterization import StiefelHouseholderCanonical, StiefelHouseholder
from src.utils.tensor_contraction import get_spectral_tt_operator_shapes, get_tt_contraction_fn_and_flops, \
    get_spectral_tt_operator_equation, compute_contraction_fn


class TestTTModules(unittest.TestCase):

    @staticmethod
    def create_factory(param, *args, **kwargs):
        return {
            'sttp': SpectralTensorsFactorySTTP,
            'svdp': SpectralTensorsFactorySVDP,
        }[param](StiefelHouseholder, StiefelHouseholderCanonical, *args, **kwargs)

    def _add_one(self, param, max_rank, shape, seed=2020):
        tf = self.create_factory(param, max_rank, init_seed=seed, init_mode='qr_randn')
        tf.add_tensor('weight', shape)
        tf.instantiate()
        m = tf()[0]
        m = m.reshape(m.shape[0], -1)
        sigma = m.svd()[1][0].item()
        self.assertAlmostEqual(sigma, 1.0, 4)

    def _test_one(self, param):
        self._add_one(param, 8, (64, 128, 3, 3))
        self._add_one(param, 128, (64, 128, 3, 3))
        self._add_one(param, 8, (64, 128, 1, 1))
        self._add_one(param, 128, (64, 128, 1, 1))
        self._add_one(param, 8, (3, 64, 3, 3))
        self._add_one(param, 128, (3, 64, 3, 3))
        self._add_one(param, 8, (3, 64, 1, 1))
        self._add_one(param, 128, (3, 64, 1, 1))

    def test_one_tt(self):
        self._test_one('sttp')

    def test_one_svd(self):
        self._test_one('svdp')

    def _test_many(self, param):
        tf = self.create_factory(param, 32, init_seed=2020, init_mode='qr_randn')
        tf.add_tensor('weight_1', (64, 128, 3, 3))
        tf.add_tensor('weight_2', (64, 128, 1, 1))
        tf.add_tensor('weight_3', (64, 64, 3, 3))
        tf.add_tensor('weight_4', (64, 64, 1, 1))
        tf.add_tensor('weight_5', (3, 64, 3, 3))
        tf.add_tensor('weight_6', (3, 64, 1, 1))
        tf.add_tensor('weight_7', (3, 64, 7, 7))
        tf.add_tensor('weight_8', (3, 64, 8, 8))
        tf.instantiate()
        weights = tf()
        for i, w in enumerate(weights):
            w = w.reshape(w.shape[0], -1)
            sigma = w.svd()[1][0].item()
            self.assertAlmostEqual(sigma, 1.0, 4, msg=i)

    def test_many_tt(self):
        self._test_many('sttp')

    def test_many_svd(self):
        self._test_many('svdp')

    def test_unidim(self):
        tf = self.create_factory('svdp', 32, init_seed=2020, init_mode='qr_randn')
        n_param_1_32 = tf.add_tensor('unidim_1_32', (1, 32))
        n_param_32_1 = tf.add_tensor('unidim_32_1', (32, 1))
        n_param_1_1 = tf.add_tensor('unidim_1_1', (1, 1))
        self.assertEqual(n_param_1_32, 32)
        self.assertEqual(n_param_32_1, 32)
        self.assertEqual(n_param_1_1, 1)
        tf.instantiate()
        tf.forward_2()
        w_1_32 = tf.get_tensor_by_name('unidim_1_32')
        w_32_1 = tf.get_tensor_by_name('unidim_32_1')
        w_1_1 = tf.get_tensor_by_name('unidim_1_1')
        self.assertAlmostEqual(w_1_32.mm(w_1_32.T).item(), 1.0, 4)
        self.assertAlmostEqual(w_32_1.T.mm(w_32_1).item(), 1.0, 4)
        self.assertAlmostEqual(w_1_1.item(), 1.0, 4)

    def _test_conv2d(self, param):
        c_orig = torch.nn.Conv2d(16, 32, 3)
        tf = self.create_factory(param, 32, init_seed=2020, init_mode='qr_randn')
        c_tf = SpectralConvNd(c_orig, "c_orig", tf)
        tf.instantiate()

        tensors = tf()
        tf.set_tensors(tensors)
        with torch.no_grad():
            c_orig.weight.copy_(tf.get_tensor_by_name("c_orig"))

        dummy = torch.randn(2, 16, 24, 32)
        out_orig = c_orig(dummy)
        out_tf = c_tf(dummy)

        self.assertEqual(out_orig.shape, out_tf.shape)
        self.assertAlmostEqual((out_orig - out_tf).abs().max().item(), 0)

    def test_conv2d_tt(self):
        self._test_conv2d('sttp')

    def test_conv2d_svd(self):
        self._test_conv2d('svdp')

    def _test_transposed_conv2d(self, param):
        c_orig = torch.nn.ConvTranspose2d(16, 32, 3)
        tf = self.create_factory(param, 32, init_seed=2020, init_mode='qr_randn')
        c_tf = SpectralConvNd(c_orig, "c_orig", tf)
        tf.instantiate()

        tensors = tf()
        tf.set_tensors(tensors)
        with torch.no_grad():
            c_orig.weight.copy_(tf.get_tensor_by_name("c_orig"))

        dummy = torch.randn(2, 16, 24, 32)
        out_orig = c_orig(dummy)
        out_tf = c_tf(dummy)

        self.assertEqual(out_orig.shape, out_tf.shape)
        self.assertAlmostEqual((out_orig - out_tf).abs().max().item(), 0)

    def test_transposed_conv2d_tt(self):
        self._test_transposed_conv2d('sttp')

    def test_transposed_conv2d_svd(self):
        self._test_transposed_conv2d('svdp')

    def _test_ttconv(self, batch_size, c_in, c_out, kernel_size, rank, in_h, in_w):
        assert kernel_size in (1, 3)
        assert type(rank) is int and 128 > rank > 0
        assert type(batch_size) is int and batch_size > 0
        assert type(c_in) is int and type(c_out) is int and c_in > 0 and c_out > 0
        assert type(in_h) is int and type(in_w) is int and in_h > 0 and in_w > 0

        import numpy as np

        rng = np.random.RandomState(2020)

        A_shape = (c_out, c_in * kernel_size * kernel_size)
        x_shape = (c_in * kernel_size * kernel_size, in_h * in_w * batch_size)

        A_tt_shapes, x_tt_shape = get_spectral_tt_operator_shapes(A_shape, x_shape, rank)

        A_tt_contraction_fn, A_tt_contraction_flops = get_tt_contraction_fn_and_flops(A_tt_shapes)

        A_tt_x_tt_equation = get_spectral_tt_operator_equation(A_tt_shapes, x_tt_shape)
        A_tt_x_tt_fn = compute_contraction_fn(A_tt_x_tt_equation, A_tt_shapes + [x_tt_shape])

        A_tt = [torch.from_numpy(rng.randn(*a)) for a in A_tt_shapes]
        x_tt = torch.from_numpy(rng.randn(*x_tt_shape))

        A = A_tt_contraction_fn(*A_tt).reshape(A_shape)
        x = x_tt.reshape(x_shape)

        Ax = A.mm(x)

        A_tt_x_tt = A_tt_x_tt_fn(*A_tt, x_tt).reshape(A_shape[0], x_shape[1])

        residual = (Ax - A_tt_x_tt).abs()
        L_inf = residual.max().item()
        RMSE = (residual ** 2).mean().sqrt()

        self.assertLess(L_inf, 0.1)
        self.assertLess(RMSE, 0.01)

    def test_ttconv(self):
        self._test_ttconv(batch_size=8, c_in=256, c_out=512, kernel_size=3, rank=32, in_h=28, in_w=28)
        self._test_ttconv(batch_size=8, c_in=128, c_out=256, kernel_size=1, rank=32, in_h=56, in_w=56)


if __name__ == '__main__':
    unittest.main()
