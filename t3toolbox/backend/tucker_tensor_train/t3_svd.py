# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.tucker_tensor_train.t3_operations as ragged_ops
import t3toolbox.backend.tucker_tensor_train.t3_orthogonalization as ragged_orth
import t3toolbox.backend.orthogonalization as orth
import t3toolbox.backend.linalg as linalg
from t3toolbox.backend.common import *

__all__ = [
    't3svd',
]

def t3svd(
        x: typ.Tuple[
            typ.Tuple[NDArray,...], # tucker_cores
            typ.Tuple[NDArray,...], # tt_cores
        ],
        min_tt_ranks:       typ.Sequence[int] = None, # len=d+1
        min_tucker_ranks:   typ.Sequence[int] = None,  # len=d
        max_tt_ranks:       typ.Sequence[int] = None, # len=d+1
        max_tucker_ranks:   typ.Sequence[int] = None, # len=d
        rtol: float = None,
        atol: float = None,
        squash_tails_first: bool = True,
) -> typ.Tuple[
    typ.Tuple[
        typ.Tuple[NDArray, ...],  # new_tucker_cores
        typ.Tuple[NDArray, ...],  # new_tt_cores
    ],
    typ.Tuple[NDArray,...], # Tucker singular values, len=d
    typ.Tuple[NDArray,...], # TT singular values, len=d+1
]:
    '''Compute (truncated) T3-SVD of TuckerTensorTrain.
    '''
    num_cores = len(x[0])

    # print('0. [B.shape for B in x[0]]=', [B.shape for B in x[0]])
    # print('0. [G.shape for G in x[1]]=', [G.shape for G in x[1]])

    # make leading and trailing TT-ranks equal to 1
    if squash_tails_first:
        x = (x[0], ragged_ops.squash_tt_tails(x[1]))

    # print('1. [B.shape for B in x[0]]=', [B.shape for B in x[0]])
    # print('1. [G.shape for G in x[1]]=', [G.shape for G in x[1]])

    # Orthogonalize Tucker matrices
    x = ragged_orth.down_orthogonalize_tucker_cores(x)

    # print('2. [B.shape for B in x[0]]=', [B.shape for B in x[0]])
    # print('2. [G.shape for G in x[1]]=', [G.shape for G in x[1]])

    # Right orthogonalize
    x = (x[0], orth.right_orthogonalize_tt_cores(x[1]))

    # print('3. [B.shape for B in x[0]]=', [B.shape for B in x[0]])
    # print('3. [G.shape for G in x[1]]=', [G.shape for G in x[1]])

    G0 = x[1][0]
    _, ss_first, _ = linalg.right_svd(G0)

    # Sweep left to right computing SVDS
    all_ss_tucker = []
    all_ss_tt = [ss_first]
    for ii in range(num_cores):
        min_rank = min_tucker_ranks[ii] if min_tucker_ranks is not None else None
        max_rank = max_tucker_ranks[ii] if max_tucker_ranks is not None else None
        # SVD inbetween TT core and Tucker core
        x, ss_tucker = ragged_orth.down_svd_tt_core(
            x, ii,
            min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol,
        )
        all_ss_tucker.append(ss_tucker)

        # print('4. [B.shape for B in x[0]]=', [B.shape for B in x[0]])
        # print('4. [G.shape for G in x[1]]=', [G.shape for G in x[1]])

        if ii < num_cores-1:
            min_rank = min_tt_ranks[ii+1] if min_tt_ranks is not None else None
            max_rank = max_tt_ranks[ii+1] if max_tt_ranks is not None else None
            # SVD inbetween ith tt core and (i+1)th tt core
            x, ss_tt = ragged_orth.left_svd_tt_core(
                x, ii,
                min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol,
            )

            # print('5. [B.shape for B in x[0]]=', [B.shape for B in x[0]])
            # print('5. [G.shape for G in x[1]]=', [G.shape for G in x[1]])

        else:
            Gf = x[1][-1]
            _, ss_tt, _ = linalg.left_svd(Gf)
        all_ss_tt.append(ss_tt)

    # print('6. [B.shape for B in x[0]]=', [B.shape for B in x[0]])
    # print('6. [G.shape for G in x[1]]=', [G.shape for G in x[1]])

    return x, tuple(all_ss_tucker), tuple(all_ss_tt)

