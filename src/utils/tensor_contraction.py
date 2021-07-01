import copy

import opt_einsum
import torch

from src.utils.tt_utils import get_ranks_tt, dim_factorize_simple


def compute_tt_contraction_equation(num_cores):
    assert num_cores > 0
    next_sym = 0

    def get_next_symbol():
        nonlocal next_sym
        s = opt_einsum.get_symbol(next_sym)
        next_sym += 1
        return s

    equation_left = ""
    equation_right = ""
    letter_core_last_rank_right = None
    for i in range(num_cores):
        letter_rank_left, letter_rank_right = None, None
        if i != 0:
            letter_rank_left = letter_core_last_rank_right
        letter_mode = get_next_symbol()
        if i != num_cores - 1:
            letter_rank_right = get_next_symbol()
            letter_core_last_rank_right = letter_rank_right
        if i > 0:
            equation_left += ','
        if letter_rank_left is not None:
            equation_left += letter_rank_left
        equation_left += letter_mode
        if letter_rank_right is not None:
            equation_left += letter_rank_right
        equation_right += letter_mode
    equation = equation_left + '->' + equation_right
    return equation


def compute_contraction_fn(equation, core_shapes):
    optimizer = 'dp'  # produces same results as opt_einsum.BranchBound(nbranch=3) but faster
    contraction_fn = opt_einsum.contract_expression(equation, *core_shapes, optimize=optimizer)
    return contraction_fn


def compute_contraction_flops(equation, core_shapes):
    optimizer = 'dp'  # produces same results as opt_einsum.BranchBound(nbranch=3) but faster
    _, path_desc = opt_einsum.contract_path(
        equation, *(torch.zeros(a) for a in core_shapes), optimize=optimizer
    )
    return int(path_desc.opt_cost)


def get_tt_contraction_fn_and_flops(core_shapes):
    if len(core_shapes[0]) == 2:
        assert len(core_shapes[-1]) == 2
        assert all(len(a) == 3 for a in core_shapes[1:-1])
    else:
        assert core_shapes[0][0] == 1 and core_shapes[-1][-1] == 1 and all(len(a) == 3 for a in core_shapes)
        core_shapes = copy.deepcopy(core_shapes)
        core_shapes[0], core_shapes[-1] = core_shapes[0][1:], core_shapes[-1][:-1]

    equation = compute_tt_contraction_equation(len(core_shapes))
    fn = compute_contraction_fn(equation, core_shapes)
    flops = compute_contraction_flops(equation, core_shapes)

    def preprocess_cores_and_contract(*args):
        cores = list(args)
        if cores[0].dim() == 2:
            assert cores[-1].dim() == 2 and all(len(a.shape) == 3 for a in cores[1:-1])
        else:
            assert cores[0].shape[0] == 1 and cores[-1].shape[-1] == 1 and all(len(a.shape) == 3 for a in cores)
            cores[0] = cores[0].squeeze(0)
            cores[-1] = cores[-1].squeeze(-1)
        return fn(*cores)

    return preprocess_cores_and_contract, flops


def get_full_operator_flops(A_shape, x_shape):
    assert len(A_shape) == 2 and len(x_shape) == 2 and A_shape[1] == x_shape[0]
    equation = 'ab,bc->ac'
    flops = compute_contraction_flops(equation, (A_shape, x_shape))
    return flops


def get_spectral_tt_operator_shapes(A_shape, x_shape, max_rank, dim_factorize_fn=dim_factorize_simple):
    assert len(A_shape) == 2 and len(x_shape) == 2 and A_shape[1] == x_shape[0]

    dim_dst_modes = dim_factorize_fn(A_shape[0], True)
    dim_src_modes = dim_factorize_fn(A_shape[1], False)
    A_modes = dim_dst_modes + dim_src_modes

    A_tt_ranks = get_ranks_tt(dim_dst_modes + dim_src_modes, max_rank)
    A_tt_shapes = list((A_tt_ranks[i], A_modes[i], A_tt_ranks[i+1]) for i in range(len(A_modes)))
    A_tt_shapes[0] = A_tt_shapes[0][1:]
    A_tt_shapes[-1] = A_tt_shapes[-1][:-1]
    x_tt_shape = dim_src_modes + [x_shape[1]]
    return A_tt_shapes, x_tt_shape


def get_spectral_tt_operator_equation(A_tt_shapes, x_tt_shape):
    assert len(A_tt_shapes) >= len(x_tt_shape) >= 2
    # A: [(o1, r1), (r1, o2, r2), (r2, i1, r3), (r3, i2, r4), (r4, i3)]
    # x: [i1, i2, i3, x1]
    num_out_modes = len(A_tt_shapes) - len(x_tt_shape) + 1
    A_in_modes = list(rmr[1] for rmr in A_tt_shapes[num_out_modes:])
    assert A_in_modes == x_tt_shape[:-1]
    num_cores = len(A_tt_shapes)

    next_sym = 0

    def get_next_symbol():
        nonlocal next_sym
        s = opt_einsum.get_symbol(next_sym)
        next_sym += 1
        return s

    equation_left = ""
    equation_right = ""
    equation_part_x = ""
    letter_core_last_rank_right = None
    for i in range(num_cores):
        is_core_right_of_sv = i >= num_out_modes
        letter_rank_left, letter_rank_right = None, None
        if i != 0:
            letter_rank_left = letter_core_last_rank_right
        letter_mode = get_next_symbol()
        if i != num_cores - 1:
            letter_rank_right = get_next_symbol()
            letter_core_last_rank_right = letter_rank_right
        if i > 0:
            equation_left += ','
        if letter_rank_left is not None:
            equation_left += letter_rank_left
        equation_left += letter_mode
        if letter_rank_right is not None:
            equation_left += letter_rank_right
        if is_core_right_of_sv:
            equation_part_x += letter_mode
        else:
            equation_right += letter_mode
    letter_x_dim1 = get_next_symbol()
    equation = equation_left + ',' + equation_part_x + letter_x_dim1 + '->' + equation_right + letter_x_dim1
    return equation
