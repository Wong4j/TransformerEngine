# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Test TE operators"""

import pytest
import paddle
from utils import assert_allclose, create_fp8_meta

from transformer_engine.paddle.cpp_extensions import (
    cast_to_fp8,
    cast_from_fp8,
    gemm,
    fp8_gemm,
    layernorm_fwd_fp8,
    layernorm_fwd,
    layernorm_bwd,
    rmsnorm_fwd_fp8,
    rmsnorm_fwd,
    rmsnorm_bwd,
)
from transformer_engine.paddle.fp8 import is_fp8_available
import transformer_engine_paddle as tex    # pylint: disable=wrong-import-order

paddle.seed(10)
GEMM_CASES = [(256, 256, 512), (32, 32, 32), (16384, 1024, 2816), (16384, 2816, 1024),
              (16384, 1024, 1024)]
is_fp8_supported, reason = is_fp8_available()


def test_quantize_dequantize():
    """
    Test cast_to_fp8 and cast_from_fp8
    """
    a = paddle.rand(shape=(32, 32), dtype='float32')
    # Init fp8_meta
    fp8_meta = create_fp8_meta(num_fp8_tensors=3, amax_history_len=10)
    for fp8_dtype in [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]:
        a_fp8 = cast_to_fp8(a, fp8_meta, tex.FP8FwdTensors.GEMM1_OUTPUT, otype=fp8_dtype)
        b = cast_from_fp8(a_fp8,
                          fp8_meta,
                          tex.FP8FwdTensors.GEMM1_OUTPUT,
                          itype=fp8_dtype,
                          otype=tex.DType.kFloat32)
        assert_allclose(a, b, rtol=5e-2, atol=5e-2)


class TestGemm:
    """
    Tests for gemm(cuBLASLt) operator
    """

    @staticmethod
    @pytest.mark.skipif(paddle.device.cuda.get_device_capability() < (8, 0),
                        reason="BF16 GEMM requires Ampere+ GPU")
    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
    def test_bf16(m, n, k):
        """
        Test "TN" BF16 GEMM
        """
        a = paddle.rand(shape=(m, k), dtype='bfloat16')
        b = paddle.rand(shape=(n, k), dtype='bfloat16')

        workspace = paddle.zeros(shape=[33_554_432], dtype='uint8')

        ref_out = paddle.matmul(a, b.T)
        # CublasLt inside tex.te_gemm assumes inputs are column major.
        # Mathematically, A@B=C is equivalent to B^T@A^T=C^T, where X^T is the
        # transpose of X.
        # Here we perform "TN" GEMM in column major, i.e., b@a^T = C^T,
        # which is equivalent to a@b^T = C in row major.
        actual_out, _, _ = gemm(b, a, paddle.bfloat16, workspace, False, None, False, False, "TN",
                                None, None, False)

        assert_allclose(actual_out, ref_out)

    @staticmethod
    @pytest.mark.skipif(paddle.device.cuda.get_device_capability() < (8, 0),
                        reason="BF16 GEMM requires Ampere+ GPU")
    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
    def test_bf16_inplace(m, n, k):
        """
        Test "TN" BF16 GEMM, with accumulate=True
        """
        min_val = -16
        max_val = 16
        a = paddle.rand(shape=(m, k), dtype='bfloat16')
        b = paddle.rand(shape=(n, k), dtype='bfloat16')
        c = paddle.cast(paddle.randint(min_val, max_val, shape=(m, n)), 'bfloat16')
        workspace = paddle.zeros(shape=[33_554_432], dtype='uint8')

        ref_out = c + paddle.matmul(a, b.T)

        actual_out = paddle.clone(c)
        _, _, _ = gemm(b, a, paddle.bfloat16, workspace, False, None, False, True, "TN", actual_out,
                       None, False)

        assert_allclose(actual_out, ref_out, rtol=5e-2, atol=5e-2)

    @staticmethod
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
    def test_fp8_randint(m, n, k):
        """
        Test "TN" FP8 GEMM
        """
        min_val = -8
        max_val = 8
        fp8_dtype = tex.DType.kFloat8E4M3
        out_dtype = paddle.float32
        fp8_meta = create_fp8_meta(num_fp8_tensors=3, amax_history_len=10)

        a = paddle.cast(paddle.randint(min_val, max_val, shape=(m, k)), 'float32')

        a_casted = cast_to_fp8(a, fp8_meta, tex.FP8FwdTensors.GEMM1_INPUT, otype=fp8_dtype)
        b = paddle.cast(paddle.randint(min_val, max_val, shape=(n, k)), 'float32')
        b_casted = cast_to_fp8(b, fp8_meta, tex.FP8FwdTensors.GEMM1_WEIGHT, otype=fp8_dtype)
        workspace = paddle.zeros(shape=[33_554_432], dtype='uint8')

        ref_out = paddle.matmul(a, b.T)
        actual_out = fp8_gemm(b_casted, fp8_meta.scale_inv, tex.FP8FwdTensors.GEMM1_WEIGHT,
                              fp8_dtype, a_casted, fp8_meta.scale_inv,
                              tex.FP8FwdTensors.GEMM1_INPUT, fp8_dtype, out_dtype, workspace)

        assert_allclose(actual_out, ref_out)


class TestLayerNorm:
    """
    Test layernorm operators
    """

    @staticmethod
    def calc_fwd_ref(x, eps, gamma, beta):
        """
        Calculate reference using paddle layer_norm op
        """
        y = paddle.nn.functional.layer_norm(x=x,
                                            normalized_shape=x.shape[1:],
                                            weight=gamma,
                                            bias=beta,
                                            epsilon=eps)
        mean = paddle.mean(x, axis=-1)
        var = paddle.var(x, axis=-1)
        inv_var = paddle.sqrt(1. / var)
        return y, mean, inv_var

    @staticmethod
    def calc_bwd_ref(x, eps, gamma, beta, dy):
        """
        Calculate reference using paddle layer_norm op
        """
        x.stop_gradient = False
        gamma.stop_gradient = False
        beta.stop_gradient = False

        y = paddle.nn.functional.layer_norm(x=x,
                                            normalized_shape=x.shape[1:],
                                            weight=gamma,
                                            bias=beta,
                                            epsilon=eps)

        paddle.autograd.backward([y], [dy], True)

        return x.grad, gamma.grad, beta.grad

    def test_layernorm_fwd(self):
        """
        Test BF16 LayerNorm Forward
        """
        N, H = (16, 32)
        eps = 1e-3
        x = paddle.uniform(shape=(N, H), dtype='bfloat16')
        gamma = paddle.uniform(shape=(H,), dtype='bfloat16')
        beta = paddle.uniform(shape=(H,), dtype='bfloat16')

        y, mu, rsigma = layernorm_fwd(x, gamma, beta, eps, tex.DType.kBFloat16)

        y_ref, mu_ref, rsigma_ref = self.calc_fwd_ref(x, eps, gamma, beta)

        assert_allclose(y, y_ref, rtol=1e-5, atol=1e-5)
        assert_allclose(mu, mu_ref, rtol=1e-3, atol=1e-3)
        assert_allclose(rsigma, rsigma_ref, rtol=5e-2, atol=5e-2)

    @staticmethod
    def test_layernorm_fwd_fp8():
        """
        Test FP8 LayerNorm Forward
        """
        fp8_dtype = tex.DType.kFloat8E4M3
        N, H = (16, 32)
        eps = 1e-3

        x = paddle.uniform(shape=(N, H), dtype='float32')
        gamma = paddle.uniform(shape=(H,), dtype='float32')
        beta = paddle.uniform(shape=(H,), dtype='float32')

        fp8_tensor = tex.FP8FwdTensors.GEMM1_INPUT
        fp8_meta = create_fp8_meta(num_fp8_tensors=1, amax_history_len=1)

        y_ref, mu_ref, rsigma_ref = layernorm_fwd(x, gamma, beta, eps, tex.DType.kFloat32)

        y_fp8, mu, rsigma = layernorm_fwd_fp8(x, gamma, beta, eps, fp8_meta, fp8_tensor, fp8_dtype)

        y = cast_from_fp8(y_fp8, fp8_meta, fp8_tensor, itype=fp8_dtype, otype=tex.DType.kFloat32)

        assert_allclose(y, y_ref, rtol=0.1, atol=0.01)
        assert_allclose(mu, mu_ref)
        assert_allclose(rsigma, rsigma_ref)

    def test_layernorm_bwd(self):
        """
        Test BF16 LayerNorm Backward
        """
        N, H = (16, 32)
        eps = 1e-3
        x = paddle.uniform(shape=(N, H), dtype='bfloat16')
        dy = paddle.uniform(shape=(N, H), dtype='bfloat16')
        gamma = paddle.uniform(shape=(H,), dtype='bfloat16')
        beta = paddle.uniform(shape=(H,), dtype='bfloat16')

        dx_ref, dgamma_ref, dbeta_ref = self.calc_bwd_ref(x, eps, gamma, beta, dy)

        _, mu, rsigma = layernorm_fwd(x, gamma, beta, eps, tex.DType.kBFloat16)
        dx, dgamma, dbeta = layernorm_bwd(dy, x, mu, rsigma, gamma)

        assert_allclose(dx, dx_ref, rtol=1e-5, atol=1e-5)
        assert_allclose(dgamma, dgamma_ref, rtol=1e-5, atol=1e-5)
        assert_allclose(dbeta, dbeta_ref, rtol=1e-5, atol=1e-5)


class TestRMSNorm:
    """
    Test rmsnorm operators
    """

    @staticmethod
    def calc_fwd_ref(x, eps, gamma):
        """
        Calculate rmsnorm reference using paddle op
        """

        norm = paddle.rsqrt(paddle.mean(x**2, axis=-1, keepdim=True) + eps)
        y = x * norm * gamma

        return y

    def calc_bwd_ref(self, x, eps, gamma, dy):
        """
        Calculate rmsnorm bwd reference using paddle op
        """
        x.stop_gradient = False
        gamma.stop_gradient = False

        y = self.calc_fwd_ref(x, eps, gamma)

        paddle.autograd.backward([y], [dy], True)

        return x.grad, gamma.grad

    def test_rmsnorm_fwd(self):
        """
        Test BF16 RMSNorm Forward
        """
        N, H = (16, 32)
        eps = 1e-3
        x = paddle.uniform(shape=(N, H), dtype='bfloat16')
        gamma = paddle.uniform(shape=(H,), dtype='bfloat16')

        y, _ = rmsnorm_fwd(x, gamma, eps, tex.DType.kBFloat16)

        y_ref = self.calc_fwd_ref(x, eps, gamma)

        assert_allclose(y, y_ref, rtol=1e-2, atol=1e-2)

    @staticmethod
    def test_rmsnorm_fwd_fp8():
        """
        Test FP8 RMSNorm Forward
        """
        fp8_dtype = tex.DType.kFloat8E4M3
        N, H = (16, 32)
        eps = 1e-3

        x = paddle.uniform(shape=(N, H), dtype='float32')
        gamma = paddle.uniform(shape=(H,), dtype='float32')

        fp8_tensor = tex.FP8FwdTensors.GEMM1_INPUT
        fp8_meta = create_fp8_meta(num_fp8_tensors=1, amax_history_len=1)

        y_ref, rsigma_ref = rmsnorm_fwd(x, gamma, eps, tex.DType.kFloat32)

        y_fp8, rsigma = rmsnorm_fwd_fp8(x, gamma, eps, fp8_meta, fp8_tensor, fp8_dtype)

        y = cast_from_fp8(y_fp8, fp8_meta, fp8_tensor, itype=fp8_dtype, otype=tex.DType.kFloat32)

        assert_allclose(y, y_ref, rtol=0.1, atol=0.01)
        assert_allclose(rsigma, rsigma_ref)

    def test_rmsnorm_bwd(self):
        """
        Test BF16 RMSNorm Backward
        """
        N, H = (16, 32)
        eps = 1e-3
        x = paddle.uniform(shape=(N, H), dtype='bfloat16')
        dy = paddle.uniform(shape=(N, H), dtype='bfloat16')
        gamma = paddle.uniform(shape=(H,), dtype='bfloat16')

        dx_ref, dgamma_ref = self.calc_bwd_ref(x, eps, gamma, dy)

        _, rsigma = rmsnorm_fwd(x, gamma, eps, tex.DType.kBFloat16)
        dx, dgamma = rmsnorm_bwd(dy, x, rsigma, gamma)

        assert_allclose(dx, dx_ref, rtol=1e-2, atol=1e-2)
        assert_allclose(dgamma, dgamma_ref, rtol=1e-2, atol=1e-2)
