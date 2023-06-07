# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""TE FP8 extensions and GEMMs"""

from typing import Optional, Tuple, Union
import paddle
import transformer_engine_paddle as tex
from .constants import TE_DType


def gemm(
    A: paddle.Tensor,
    B: paddle.Tensor,
    dtype: paddle.dtype,
    workspace: paddle.Tensor,
    gelu: bool = False,
    gelu_input: Optional[paddle.Tensor] = None,
    grad: bool = False,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[paddle.Tensor] = None,
    bias: Optional[paddle.Tensor] = None,
    use_bias: bool = False,
) -> Tuple[Union[paddle.Tensor, None], ...]:
    """Non FP8 GEMM."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"

    return_output = False
    if out is None:
        out = paddle.empty(
            shape=[
                B.shape[1] if transb else B.shape[0],
                A.shape[0] if transa else A.shape[1],
            ],
            dtype=dtype,
        )
        return_output = True

    if gelu and not grad:
        gelu_input = paddle.empty_like(out, dtype=dtype)
    elif not gelu:
        gelu_input = None

    if grad and use_bias:
        grad_bias = paddle.empty(shape=[B.shape[1]], dtype=out.dtype)
    else:
        grad_bias = None

    bias = bias if use_bias else None

    assert A.dtype == dtype and B.dtype == dtype, \
        f'Expected dtype={dtype}, but found A.dtype={A.dtype} and B.dtype={B.dtype}'
    input_dtype = TE_DType[dtype]
    output_dtype = TE_DType[out.dtype]
    if use_bias:
        bias_dtype = TE_DType[grad_bias.dtype] if grad else TE_DType[bias.dtype]
    else:
        bias_dtype = output_dtype

    tex.te_gemm(
        A,
        None,
        B,
        None,
        grad_bias if grad else bias,
        out,
        None,    # out_scale
        None,    # out_amax
        gelu_input,
        workspace,
        0,    # A_index
        0,    # B_index
        0,    # D_index
        int(input_dtype),
        int(input_dtype),
        int(output_dtype),
        int(bias_dtype),
        transa,
        transb,
        grad,
        workspace.shape[0],
        accumulate,
        False,    # use_split_accumulator
        0,    # math_sm_count
    )

    if return_output:
        return out, grad_bias, gelu_input
    return None, grad_bias, gelu_input


def fp8_gemm(
    A: paddle.Tensor,
    A_scale_inv: paddle.Tensor,
    A_fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    A_dtype: tex.DType,
    B: paddle.Tensor,
    B_scale_inv: paddle.Tensor,
    B_fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    B_dtype: tex.DType,
    out_dtype: paddle.dtype,
    workspace: paddle.Tensor,
    gelu: bool = False,
    accumulate: bool = False,
    out: Optional[paddle.Tensor] = None,
    out_index=None,
    fp8_meta_tensor: tex.FP8TensorMeta = None,
    bias: Optional[paddle.Tensor] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[tex.DType] = None,
) -> paddle.Tensor:
    """TN layout GEMM with fp8 inputs."""

    if D_dtype is not None and D_dtype in [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]:
        assert fp8_meta_tensor is not None and out_index is not None

    return_output = False
    if out is None:
        out = paddle.empty(
            shape=[
                B.shape[0],
                A.shape[0],
            ],
            dtype=out_dtype,
        )
        return_output = True
    # Use bfloat16 as default bias_dtype
    bias_dtype = paddle.bfloat16 if bias is None else bias.dtype
    if gelu:
        gelu_input = paddle.empty_like(out, dtype=bias_dtype)
    else:
        gelu_input = None
    bias_dtype = TE_DType[bias_dtype]

    out_dtype = TE_DType[out.dtype] if D_dtype is None else D_dtype

    tex.te_gemm(
        A,
        A_scale_inv,
        B,
        B_scale_inv,
        bias if use_bias else None,
        out,
        None if out_index is None else fp8_meta_tensor.scale,
        None if out_index is None else fp8_meta_tensor.amax_history,
        gelu_input,    # this is pre_gelu_out
        workspace,
        int(A_fp8_tensor),
        int(B_fp8_tensor),
        0 if out_index is None else out_index,
        int(A_dtype),
        int(B_dtype),
        int(out_dtype),
        int(bias_dtype),
        True,    # transa
        False,    # transb
        False,    # grad
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
        0,    # math_sm_count
    )

    if return_output:
        if gelu:
            return out, gelu_input
        return out
    if gelu:
        return gelu_input
    return None


def cast_to_fp8(
    inp: paddle.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> paddle.Tensor:
    """Cast input to FP8"""
    out, _, _ = tex.cast_to_fp8(
        inp,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        fp8_meta_tensor.scale_inv,
        int(fp8_tensor),
        int(otype),
    )
    return out


def cast_from_fp8(
    inp: paddle.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    itype: tex.DType,
    otype: tex.DType,
) -> paddle.Tensor:
    """Cast input from FP8"""
    return tex.cast_from_fp8(
        inp,
        fp8_meta_tensor.scale_inv,
        int(fp8_tensor),
        int(itype),
        int(otype),
    )


def transpose(
    inp: paddle.Tensor,
    otype: tex.DType,
) -> paddle.Tensor:
    """Transpose input"""
    return tex.te_transpose(
        inp,
        int(otype),
    )


def cast_transpose(
    inp: paddle.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> Union[Tuple[paddle.Tensor, paddle.Tensor], None]:
    """Cast + Transpose with FP8 output"""
    cast_out, transpose_out, _, _ = tex.te_cast_transpose(
        inp,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        fp8_meta_tensor.scale_inv,
        int(fp8_tensor),
        int(otype),
    )

    return cast_out, transpose_out


def te_gelu(
    inp: paddle.Tensor,
    otype: tex.DType,
) -> paddle.Tensor:
    """Non FP8 GELU"""
    return tex.te_gelu(
        inp,
        int(otype),
    )


def gelu_fp8(
    inp: paddle.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> paddle.Tensor:
    """GELU + FP8 cast"""
    out, _, _ = tex.te_gelu_fp8(
        inp,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        fp8_meta_tensor.scale_inv,
        int(fp8_tensor),
        int(otype),
    )

    return out


def dgelu_cast_transpose_bgrad_fp8(
    grad_output: paddle.Tensor,
    gelu_input: paddle.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    Fused dgelu + cast / transpose / reduce the result of
    the GELU backward along the first dimension
    """
    cast_dgelu, transpose_dgelu, dbias, _, _ = tex.te_cast_transpose_bgrad_dgelu(
        grad_output,
        gelu_input,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        fp8_meta_tensor.scale_inv,
        int(fp8_tensor),
        int(otype),
    )

    return cast_dgelu, transpose_dgelu, dbias


def layernorm_fwd_fp8(
    inp: paddle.Tensor,
    weight: paddle.Tensor,
    bias: paddle.Tensor,
    eps: float,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """LayerNorm with FP8 output"""
    out, mu, rsigma, _, _ = tex.te_layernorm_fwd_fp8(inp, weight, bias, fp8_meta_tensor.scale,
                                                     fp8_meta_tensor.amax_history,
                                                     fp8_meta_tensor.scale_inv, eps,
                                                     int(fp8_tensor), int(otype), sm_margin,
                                                     zero_centered_gamma)
    return out, mu, rsigma


def layernorm_fwd(
    inp: paddle.Tensor,
    weight: paddle.Tensor,
    bias: paddle.Tensor,
    eps: float,
    otype: tex.DType,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Non-FP8 LayerNorm forward"""
    return tex.te_layernorm_fwd(inp, weight, bias, eps, int(otype), sm_margin, zero_centered_gamma)


def layernorm_bwd(
    dz: paddle.Tensor,
    x: paddle.Tensor,
    mu: paddle.Tensor,
    rsigma: paddle.Tensor,
    gamma: paddle.Tensor,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Non-FP8 LayerNorm backward"""
    return tex.te_layernorm_bwd(dz, x, mu, rsigma, gamma, sm_margin, zero_centered_gamma)


def rmsnorm_fwd(
    inp: paddle.Tensor,
    weight: paddle.Tensor,
    eps: float,
    otype: tex.DType,
    sm_margin: int = 0,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Non-FP8 RMSNorm forward"""
    return tex.te_rmsnorm_fwd(inp, weight, eps, int(otype), sm_margin)


def rmsnorm_fwd_fp8(
    inp: paddle.Tensor,
    weight: paddle.Tensor,
    eps: float,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    sm_margin: int = 0,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """RMSNorm with FP8 output"""
    out, rsigma, _, _ = tex.te_rmsnorm_fwd_fp8(inp, weight, fp8_meta_tensor.scale,
                                               fp8_meta_tensor.amax_history,
                                               fp8_meta_tensor.scale_inv, eps, int(fp8_tensor),
                                               int(otype), sm_margin)
    return out, rsigma


def rmsnorm_bwd(
    dz: paddle.Tensor,
    x: paddle.Tensor,
    rsigma: paddle.Tensor,
    gamma: paddle.Tensor,
    sm_margin: int = 0,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Non-FP8 RMSNorm backward"""
    return tex.te_rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin)
