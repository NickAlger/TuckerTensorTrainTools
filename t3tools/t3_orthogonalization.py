# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import typing as typ
from t3tools.linalg import *
from t3tools.tucker_tensor_train import *

try:
    import jax.numpy as jnp
except:
    print('jax import failed in tucker_tensor_train. Defaulting to numpy.')
    jnp = np

NDArray = typ.Union[np.ndarray, jnp.ndarray]

__all__ = [
    'T3Base',
    'T3Variation',
    'up_svd_ith_basis_core',
    'left_svd_ith_tt_core',
    'right_svd_ith_tt_core',
    'up_svd_ith_tt_core',
    'down_svd_ith_tt_core',
    'orthogonalize_relative_to_ith_basis_core',
    'orthogonalize_relative_to_ith_tt_core',
]

T3Base = typ.Tuple[
    typ.Sequence[NDArray],  # base_basis_cores. B_xo B_yo = I_xy    B.shape = (n, N)
    typ.Sequence[NDArray],  # base_left_tt_cores. P_iax P_iay = I_xy, P.shape = (rL, n, rR)
    typ.Sequence[NDArray],  # base_right_tt_cores. Q_xaj Q_yaj = I_xy  Q.shape = (rL, n, rR)
    typ.Sequence[NDArray],  # base_outer_tt_cores. R_ixj R_iyj = I_xy  R.shape = (rL, n, rR)
]

T3Variation = typ.Tuple[
    typ.Sequence[NDArray],  # variation_basis_cores.
    typ.Sequence[NDArray],  # variation_tt_cores.
]


def t3_check_base(
        orth_cores: T3Base,
) -> None:
    basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = orth_cores

    num_cores = len(basis_cores)
    all_num_cores = [len(basis_cores), len(left_tt_cores), len(right_tt_cores), len(outer_tt_cores)]
    if all_num_cores != [num_cores]*4:
        raise RuntimeError(
            'Orthogonals have different numbers of cores. These should all be equal:\n'
            + '[len(basis_cores), len(left_tt_cores), len(right_tt_cores), len(outer_tt_cores)]=\n'
            + str(all_num_cores)
        )

    # Check that basis_cores are matrices
    for ii, B in enumerate(basis_cores):
        if len(B.shape) != 2:
            raise RuntimeError(
                'basis_core is not a matrix:\n'
                + 'basis_cores['+str(ii) + '].shape=' + str(B.shape)
            )

    # Check that outer_tt_cores are 3-tensors with leading and trailing 1 dims
    for ii, G in enumerate(outer_tt_cores):
        if len(G.shape) != 3:
            raise RuntimeError(
                'outer_tt_core is not a 3-tensor:\n'
                + 'outer_tt_cores['+str(ii) + '].shape=' + str(G.shape)
            )

    if outer_tt_cores[0].shape[0] != 1:
        raise RuntimeError(
            'First outer_tt_core must have shape (1, . , .).\n'
            + 'outer_tt_cores[0].shape=' + str(outer_tt_cores[0].shape)
        )

    if outer_tt_cores[-1].shape[2] != 1:
        raise RuntimeError(
            'Last outer_tt_core must have shape ( . , . , 1).\n'
            + 'outer_tt_cores[-1].shape=' + str(outer_tt_cores[-1].shape)
        )

    # Check that left_tt_cores are 3-tensors with leading and trailing 1 dims
    for ii, G in enumerate(left_tt_cores):
        if len(G.shape) != 3:
            raise RuntimeError(
                'left_tt_core is not a 3-tensor:\n'
                + 'left_tt_cores['+str(ii) + '].shape=' + str(G.shape)
            )

    if left_tt_cores[0].shape[0] != 1:
        raise RuntimeError(
            'First left_tt_core must have shape (1, . , .).\n'
            + 'left_tt_cores[0].shape=' + str(left_tt_cores[0].shape)
        )

    if left_tt_cores[-1].shape[2] != 1:
        raise RuntimeError(
            'Last left_tt_core must have shape ( . , . , 1).\n'
            + 'left_tt_cores[-1].shape=' + str(left_tt_cores[-1].shape)
        )

    # Check that right_tt_cores are 3-tensors with leading and trailing 1 dims
    for ii, G in enumerate(right_tt_cores):
        if len(G.shape) != 3:
            raise RuntimeError(
                'right_tt_core is not a 3-tensor:\n'
                + 'right_tt_cores['+str(ii) + '].shape=' + str(G.shape)
            )

    if right_tt_cores[0].shape[0] != 1:
        raise RuntimeError(
            'First right_tt_core must have shape (1, . , .).\n'
            + 'right_tt_cores[0].shape=' + str(right_tt_cores[0].shape)
        )

    if right_tt_cores[-1].shape[2] != 1:
        raise RuntimeError(
            'Last right_tt_core must have shape ( . , . , 1).\n'
            + 'right_tt_cores[-1].shape=' + str(right_tt_cores[-1].shape)
        )

    # Check outer-left consistency
    for ii in range(1, num_cores):
        GO = outer_tt_cores[ii]
        GL = left_tt_cores[ii-1]
        if GO.shape[0] != GL.shape[2]:
            raise RuntimeError(
                'Inconsistency in outer_tt_core and left_tt_core shapes:\n'
                + 'outer_tt_cores['+str(ii)+'].shape=' + str(GO.shape) + '\n'
                + 'left_tt_cores['+str(ii-1)+'].shape=' + str(GL.shape)
            )

    # Check outer-right consistency
    for ii in range(0, num_cores-1):
        GO = outer_tt_cores[ii]
        GR = right_tt_cores[ii+1]
        if GO.shape[2] != GR.shape[0]:
            raise RuntimeError(
                'Inconsistency in outer_tt_core and right_tt_core shapes:\n'
                + 'outer_tt_cores['+str(ii)+'].shape=' + str(GO.shape) + '\n'
                + 'right_tt_cores['+str(ii+11)+'].shape=' + str(GR.shape)
            )


def t3_check_nonorthogonals_fit(
        orth_cores: T3Orthogonals,
        nonorth_cores: T3NonOrthogonals,
) -> None:
    orth_basis_cores, left_orth_tt_cores, right_orth_tt_cores, outer_orth_tt_cores = orth_cores
    nonorth_basis_cores, nonorth_tt_cores = nonorth_cores





##########################################
########    Orthogonalization    #########
##########################################

def up_svd_ith_basis_core(
        ii: int, # which base core to orthogonalize
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    NDArray, # ss_x. singular values
]:
    '''Compute SVD of ith basis core and contract non-orthogonal factor up into the TT-core above.

    Parameters
    ----------
    ii: int
        index of basis core to SVD
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but with ith basis core orthogonal.
        new_tt_cores[ii].shape = (ri, new_ni, r(i+1))
        new_basis_cores[ii].shape = (new_ni, Ni)
        new_basis_cores[ii] @ new_basis_cores[ii].T = identity matrix
    ss_x: NDArray
        Singular values of prior ith basis core. shape=(new_ni,).

    See Also
    --------
    truncated_svd
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> ind = 1
    >>> x2, ss = up_svd_ith_basis_core(ind, x)
    >>> print(np.linalg.norm(t3_to_dense(x) - t3_to_dense(x2)))
        5.772851635866132e-13
    >>> basis_cores2, tt_cores2 = x2
    >>> rank = len(ss)
    >>> B = basis_cores2[ind]
    >>> print(np.linalg.norm(B @ B.T - np.eye(rank)))
        8.456498415401757e-16
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x
    G_a_i_b = tt_cores[ii]
    U_i_o = basis_cores[ii]
    U_o_i = U_i_o.T

    U2_o_x, ss_x, Vt_x_i = truncated_svd(U_o_i, min_rank, max_rank, rtol, atol, use_jax)
    R_x_i = xnp.einsum('x,xi->xi', ss_x, Vt_x_i)
    # U2_o_x, R_x_i = xnp.linalg.qr(U_o_i, mode='reduced')

    G2_a_x_b = xnp.einsum('aib,xi->axb', G_a_i_b, R_x_i)
    U2_x_o = U2_o_x.T

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G2_a_x_b

    new_basis_cores = list(basis_cores)
    new_basis_cores[ii] = U2_x_o

    new_x = (tuple(new_basis_cores), tuple(new_tt_cores))

    return new_x, ss_x


def left_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    NDArray, # singular values, shape=(r(i+1),)
]:
    '''Compute SVD of ith TT-core left unfolding and contract non-orthogonal factor into the TT-core to the right.

    Parameters
    ----------
    ii: int
        index of TT-core to SVD
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but with ith TT-core orthogonal.
        new_tt_cores[ii].shape = (ri, ni, new_r(i+1))
        new_tt_cores[ii+1].shape = (new_r(i+1), n(i+1), r(i+2))
        einsum('iaj,iak->jk', new_tt_cores[ii], new_tt_cores[ii]) = identity matrix
    ss_x: NDArray
        Singular values of prior ith TT-core left unfolding. shape=(new_r(i+1),).

    See Also
    --------
    truncated_svd
    left_svd_3tensor
    up_svd_ith_basis_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> ind = 1
    >>> x2, ss = left_svd_ith_tt_core(ind, x)
    >>> print(np.linalg.norm(t3_to_dense(x) - t3_to_dense(x2)))
        5.186463661974644e-13
    >>> basis_cores2, tt_cores2 = x2
    >>> rank = len(ss)
    >>> G = tt_cores2[ind]
    >>> print(np.linalg.norm(jnp.einsum('iaj,iak->jk', G, G) - np.eye(rank)))
        4.453244025338311e-16
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x

    A0_a_i_b = tt_cores[ii]
    B0_b_j_c = tt_cores[ii+1]

    A_a_i_x, ss_x, Vt_x_b = left_svd_3tensor(A0_a_i_b, min_rank, max_rank, rtol, atol, use_jax)
    B_x_j_c = xnp.tensordot(ss_x.reshape((-1,1)) * Vt_x_b, B0_b_j_c, axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = A_a_i_x
    new_tt_cores[ii+1] = B_x_j_c

    return (tuple(basis_cores), tuple(new_tt_cores)), ss_x


def right_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    NDArray, # singular values, shape=(new_ri,)
]:
    '''Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor into the TT-core to the left.

    Parameters
    ----------
    ii: int
        index of TT-core to SVD
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but with ith TT-core orthogonal.
        new_tt_cores[ii].shape = (new_ri, ni, r(i+1))
        new_tt_cores[ii-1].shape = (r(i-1), n(i-1), new_ri)
        einsum('iaj,kaj->ik', new_tt_cores[ii], new_tt_cores[ii]) = identity matrix
    ss_x: NDArray
        Singular values of prior ith TT-core right unfolding. shape=(new_ri,).

    See Also
    --------
    truncated_svd
    left_svd_3tensor
    up_svd_ith_basis_core
    left_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> ind = 1
    >>> x2, ss = right_svd_ith_tt_core(ind, x)
    >>> print(np.linalg.norm(t3_to_dense(x) - t3_to_dense(x2)))
        5.304678679078675e-13
    >>> basis_cores2, tt_cores2 = x2
    >>> rank = len(ss)
    >>> G = tt_cores2[ind]
    >>> print(np.linalg.norm(jnp.einsum('iaj,kaj->ik', G, G) - np.eye(rank)))
        4.207841813173725e-16
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x

    A0_a_i_b = tt_cores[ii-1]
    B0_b_j_c = tt_cores[ii]

    U_b_x, ss_x, B_x_j_c = right_svd_3tensor(B0_b_j_c, min_rank, max_rank, rtol, atol, use_jax)
    A_a_i_x = xnp.tensordot(A0_a_i_b, U_b_x * ss_x.reshape((1,-1)), axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii-1] = A_a_i_x
    new_tt_cores[ii] = B_x_j_c

    return (tuple(basis_cores), tuple(new_tt_cores)), ss_x


def up_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    NDArray, # singular values, shape=(new_ni,)
]:
    '''Compute SVD of ith TT-core outer unfolding and keep non-orthogonal factor with this core.

    Parameters
    ----------
    ii: int
        index of TT-core to SVD
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor.
        new_tt_cores[ii].shape = (ri, new_ni, r(i+1))
        new_basis_cores[ii].shape = (new_ni, Ni)
    ss_x: NDArray
        Singular values of prior ith TT-core outer unfolding. shape=(new_ri,).

    See Also
    --------
    truncated_svd
    outer_svd_3tensor
    up_svd_ith_basis_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    down_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> x2, ss = up_svd_ith_tt_core(1, x)
    >>> print(np.linalg.norm(t3_to_dense(x) - t3_to_dense(x2)))
        1.002901486286745e-12
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x

    G0_a_i_b = tt_cores[ii]
    Q0_i_o = basis_cores[ii]

    U_a_x_b, ss_x, Vt_x_i = outer_svd_3tensor(G0_a_i_b, min_rank, max_rank, rtol, atol, use_jax)

    G_a_x_b = xnp.einsum('axb,x->axb', U_a_x_b, ss_x)
    Q_x_o = xnp.tensordot(Vt_x_i, Q0_i_o, axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G_a_x_b

    new_basis_cores = list(basis_cores)
    new_basis_cores[ii] = Q_x_o

    return (tuple(new_basis_cores), tuple(new_tt_cores)), ss_x


def down_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    NDArray, # singular values, shape=(new_ni,)
]:
    '''Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor down into the basis core below.

    Parameters
    ----------
    ii: int
        index of TT-core to SVD
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but with ith TT-core outer orthogonal.
        new_tt_cores[ii].shape = (ri, new_ni, r(i+1))
        new_basis_cores[ii].shape = (new_ni, Ni)
        einsum('iaj,ibj->ab', new_tt_cores[ii], new_tt_cores[ii]) = identity matrix
    ss_x: NDArray
        Singular values of prior ith TT-core outer unfolding. shape=(new_ni,).

    See Also
    --------
    truncated_svd
    outer_svd_3tensor
    up_svd_ith_basis_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> ind = 1
    >>> x2, ss = down_svd_ith_tt_core(ind, x)
    >>> print(np.linalg.norm(t3_to_dense(x) - t3_to_dense(x2)))
        4.367311712704942e-12
    >>> basis_cores2, tt_cores2 = x2
    >>> rank = len(ss)
    >>> G = tt_cores2[ind]
    >>> print(np.linalg.norm(jnp.einsum('iaj,ibj->ab', G, G) - np.eye(rank)))
        1.0643458053135608e-15
    '''
    basis_cores, tt_cores = x

    G0_a_i_b = tt_cores[ii]
    Q0_i_o = basis_cores[ii]

    G_a_x_b, ss_x, Vt_x_i = outer_svd_3tensor(G0_a_i_b, min_rank, max_rank, rtol, atol, use_jax)

    Q_x_o = (ss_x.reshape((-1,1)) * Vt_x_i) @ Q0_i_o

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G_a_x_b

    new_basis_cores = list(basis_cores)
    new_basis_cores[ii] = Q_x_o

    return (tuple(new_basis_cores), tuple(new_tt_cores)), ss_x


def orthogonalize_relative_to_ith_basis_core(
        ii: int,
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> TuckerTensorTrain:
    '''Orthogonalize all cores in the TuckerTensorTrain except for the ith basis core.

    Orthogonal is done relative to the ith basis core:
        - ith basis core is not orthogonalized
        - All other basis cores are orthogonalized.
        - TT-cores to the left are left orthogonalized.
        - TT-core directly above is outer orthogonalized.
        - TT-cores to the right are right orthogonalized.

    Parameters
    ----------
    ii: int
        index of basis core that is not orthogonalized
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but orthogonalized relative to the ith basis core.

    See Also
    --------
    up_svd_ith_basis_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> x2 = orthogonalize_relative_to_ith_basis_core(1, x)
    >>> print(np.linalg.norm(t3_to_dense(x) - t3_to_dense(x2)))
        8.800032152216517e-13
    >>> ((B0, B1, B2), (G0, G1, G2)) = x2
    >>> X = jnp.einsum('xi,axb,byc,czd,zk->iyk', B0, G0, G1, G2, B2) # Contraction of everything except B1
    >>> print(np.linalg.norm(jnp.einsum('iyk,iwk->yw', X, X) - np.eye(B1.shape[0])))
        1.7116160385376214e-15
    '''
    shape, tucker_ranks, tt_ranks = t3_structure(x)

    new_x = x
    for jj in range(ii):
        new_x = down_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = up_svd_ith_basis_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = left_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]

    for jj in range(len(shape)-1, ii, -1):
        new_x = down_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = up_svd_ith_basis_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = right_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]

    new_x = down_svd_ith_tt_core(ii, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
    return new_x


def orthogonalize_relative_to_ith_tt_core(
        ii: int,
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> TuckerTensorTrain:
    '''Orthogonalize all cores in the TuckerTensorTrain except for the ith TT-core.

    Orthogonal is done relative to the ith TT-core:
        - All basis cores are orthogonalized.
        - TT-cores to the left are left orthogonalized.
        - ith TT-core is not orthogonalized.
        - TT-cores to the right are right orthogonalized.

    Parameters
    ----------
    ii: int
        index of TT-core that is not orthogonalized
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    See Also
    --------
    up_svd_ith_basis_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but orthogonalized relative to the ith TT-core.

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_orthogonalization_and_svd import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> x2 = orthogonalize_relative_to_ith_tt_core(1, x)
    >>> print(np.linalg.norm(t3_to_dense(x) - t3_to_dense(x2)))
        8.800032152216517e-13
    >>> ((B0, B1, B2), (G0, G1, G2)) = x2
    >>> XL = np.einsum('axb,xi -> aib', G0, B0) # Everything to the left of G1
    >>> print(np.linalg.norm(np.einsum('aib,aic->bc', XL, XL) - np.eye(G1.shape[0])))
        9.820411604510197e-16
    >>> print(np.linalg.norm(np.einsum('xi,yi->xy', B1, B1) - np.eye(G1.shape[1]))) # Everything below G1
        2.1875310121178e-15
    >>> XR = np.einsum('axb,xi->aib', G2, B2) # Everything to the right of G1
    >>> print(np.linalg.norm(np.einsum('aib,cib->ac', XR, XR) - np.eye(G1.shape[2])))
        1.180550381921849e-15
    '''
    shape, tucker_ranks, tt_ranks = t3_structure(x)

    new_x = x
    for jj in range(ii):
        new_x = down_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = up_svd_ith_basis_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = left_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]

    for jj in range(len(shape)-1, ii, -1):
        new_x = down_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = up_svd_ith_basis_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = right_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]

    new_x = up_svd_ith_basis_core(ii, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
    return new_x


def t3_orthogonals(
        x: TuckerTensorTrain,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # non-orthogonal cores
    T3Orthogonals, # orthogonalizations of x
]:
    t3_check(x)

    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x
    num_cores = len(tt_cores)

    # Orthogonalize basis matrices
    for ii in range(num_cores):
        x = up_svd_ith_basis_core(ii, x, use_jax=use_jax)[0]
    orthogonal_basis_cores = tuple([B.copy() for B in x[0]])

    # Right orthogonalize
    for ii in range(num_cores-1, 0, -1): # num_cores-1, num_cores-2, ..., 1
        x = right_svd_ith_tt_core(ii, x, use_jax=use_jax)[0]
    right_orthogonal_tt_cores = tuple([G.copy() for G in x[1]])

    non_orthogonal_basis_cores = []
    non_orthogonal_tt_cores = []

    left_orthogonal_tt_cores = []
    outer_orthogonal_tt_cores = []
    # Sweep left to right
    for ii in range(num_cores):
        # SVD inbetween tt core and basis core
        x, ss_basis = up_svd_ith_tt_core(
            ii, x, use_jax=use_jax,
        )
        all_ss_basis.append(ss_basis)

        if ii < num_cores-1:
            # SVD inbetween ith tt core and (i+1)th tt core
            x, ss_tt = left_svd_ith_tt_core(
                ii, x, use_jax=use_jax,
            )
        else:
            Gf = x[1][-1]
            _, ss_tt, _ = left_svd_3tensor(Gf, use_jax=use_jax)
        all_ss_tt.append(ss_tt)

    return x, tuple(all_ss_basis), tuple(all_ss_tt)


