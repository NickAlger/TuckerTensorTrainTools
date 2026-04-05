# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import typing as typ

try:
    import jax.numpy as jnp
except:
    print('jax import failed in tucker_tensor_train. Defaulting to numpy.')
    jnp = np

from t3tools.tucker_tensor_train import *
from t3tools.t3_orthogonalization import *

__all__ = [
    't3_add',
    't3_scale',
    't3_neg',
    't3_sub',
    't3_dot_t3',
    't3_norm',
]


#############################################################################################
########    TuckerTensorTrain Linear algebra operations, inner product, and norm    #########
#############################################################################################

def t3_add(
        x: TuckerTensorTrain,
        y: TuckerTensorTrain,
        use_jax: bool = False,
) -> TuckerTensorTrain:
    """Add TuckerTensorTrains x, y with the same shape, yielding a TuckerTensorTrain z=x+y with summed ranks.

    Parameters
    ----------
    x: TuckerTensorTrain
        First summand. structure=((N1,...,Nd), (n1,...,nd), (1, r1,...,r(d-1),1))
    y: TuckerTensorTrain
        Second summand. structure=((N1,...,Nd), (m1,...,md), (1, q1,...,q(d-1),1))
    use_jax: bool
        If True, returned TuckerTensorTrain cores are jnp.ndaray. Otherwise, np.ndarray. Default: False

    Returns
    -------
    z: TuckerTensorTrain
        Sum of Tucker tensor trains, z=x+y. structure=((N1,...,Nd), (n1+m1,...,nd+md), (1, r1+q1,...,r(d-1)+q(d-1),1))

    Raises
    ------
    RuntimeError
        - Error raised if either of the TuckerTensorTrains are internally inconsistent
        - Error raised if the TuckerTensorTrains have different shapes.

    See Also
    --------
    TuckerTensorTrain
    t3_shape
    t3_scale
    t3_sub
    t3_neg

    Notes
    -----
    The basis cores for z are vertically stacked versions of the basis cores for x and y

    The first TT-core for z is a block 1x2x2 tensor, with:
        - the first TT-core for x in the (0,0,0) block,
        - the first TT-core for y in the (0,1,1) block,
        - zeros elsewhere

    Intermediate TT-cores for z are block 2x2x2 tensors, with:
        - the TT-cores for x in the (0,0,0) block,
        - the TT-cores for y in the (1,1,1) block,
        - zeros elsewhere.

    The last TT-cores for z is a block 2x2x1 tensor, with:
        - the last TT-core for x in the (0,0,0) block,
        - the last TT-core for y in the (1,1,0) block,
        - zeros elsewhere

    Examples
    --------
    >>> import numpy as np
    >>> from t3tools import *
    >>> x = t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> y = t3_corewise_randn(((14,15,16), (3,7,2), (1,5,6,1)))
    >>> z = t3_add(x, y)
    >>> print(t3_structure(z))
    ((14, 15, 16), (7, 12, 8), (1, 8, 8, 1))
    >>> print(np.linalg.norm(t3_to_dense(x) + t3_to_dense(y) - t3_to_dense(z)))
    6.524094086845177e-13
    """
    t3_check(x)
    t3_check(y)

    x_shape = t3_structure(x)[0]
    y_shape = t3_structure(y)[0]
    if x_shape != y_shape:
        raise RuntimeError(
            'Attempted to add TuckerTensorTrains x+y with inconsistent shapes.'
            + str(x_shape) + ' = x_shape != y_shape = ' + str(y_shape)
        )

    xnp = jnp if use_jax else np

    basis_cores_x, tt_cores_x = x
    basis_cores_y, tt_cores_y = y
    basis_cores_z = [xnp.concatenate([Bx, By], axis=0) for Bx, By in zip(basis_cores_x, basis_cores_y)]

    tt_cores_z = []

    Gx = tt_cores_x[0]
    Gy = tt_cores_y[0]
    G000 = Gx
    G001 = xnp.zeros((1, Gx.shape[1], Gy.shape[2]))
    G010 = xnp.zeros((1, Gy.shape[1], Gx.shape[2]))
    G011 = Gy
    Gz = xnp.block([[G000, G001], [G010, G011]])
    tt_cores_z.append(Gz)

    for Gx, Gy in zip(tt_cores_x[1:-1], tt_cores_y[1:-1]):
        G000 = Gx
        G001 = xnp.zeros((Gx.shape[0], Gx.shape[1], Gy.shape[2]))
        G010 = xnp.zeros((Gx.shape[0], Gy.shape[1], Gx.shape[2]))
        G011 = xnp.zeros((Gx.shape[0], Gy.shape[1], Gy.shape[2]))
        G100 = xnp.zeros((Gy.shape[0], Gx.shape[1], Gx.shape[2]))
        G101 = xnp.zeros((Gy.shape[0], Gx.shape[1], Gy.shape[2]))
        G110 = xnp.zeros((Gy.shape[0], Gy.shape[1], Gx.shape[2]))
        G111 = Gy
        Gz = xnp.block([[[G000, G001], [G010, G011]], [[G100, G101], [G110, G111]]])
        tt_cores_z.append(Gz)

    Gx = tt_cores_x[-1]
    Gy = tt_cores_y[-1]
    G000 = Gx.transpose([2,0,1])
    G001 = xnp.zeros((1, Gx.shape[0], Gy.shape[1]))
    G010 = xnp.zeros((1, Gy.shape[0], Gx.shape[1]))
    G011 = Gy.transpose([2,0,1])
    Gz = xnp.block([
        [G000, G001],
        [G010, G011],
    ]).transpose([1,2,0])
    tt_cores_z.append(Gz)

    z = (tuple(basis_cores_z), tuple(tt_cores_z))
    return z


def t3_scale(
        x: TuckerTensorTrain,
        s, # scalar
) -> TuckerTensorTrain:
    """Scale TuckerTensorTrain x by a scaling factor s.

    Parameters
    ----------
    x: TuckerTensorTrain
        Tucker tensor train
    s: scalar
        scaling factor

    Returns
    -------
    z: TuckerTensorTrain
        scaled TuckerTensorTrain z=s*x, with the same structure as x.

    Raises
    ------
    RuntimeError
        - Error raised if the TuckerTensorTrains are internally inconsistent

    See Also
    --------
    TuckerTensorTrain
    t3_add
    t3_neg
    t3_sub

    Notes
    -----
    Scales the last basis core of x, leaving all other cores unchanged

    Examples
    --------
    >>> import numpy as np
    >>> from t3tools import *
    >>> x = t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> s = 3.2
    >>> z = t3_scale(x, s)
    >>> print(np.linalg.norm(s*t3_to_dense(x) - t3_to_dense(z)))
    1.6268482531988893e-13
    """
    t3_check(x)

    basis_cores, tt_cores = x

    scaled_basis_cores = [B.copy() for B in basis_cores]
    scaled_basis_cores[-1] = s*scaled_basis_cores[-1]

    copied_tt_cores = [G.copy() for G in tt_cores]

    z = (tuple(scaled_basis_cores), tuple(copied_tt_cores))
    return z


def t3_neg(
        x: TuckerTensorTrain,
) -> TuckerTensorTrain:
    """Scale TuckerTensorTrain x by a scaling factor -1.

    Parameters
    ----------
    x: TuckerTensorTrain
        Tucker tensor train

    Returns
    -------
    TuckerTensorTrain
        scaled TuckerTensorTrain -x, with the same structure as x.

        Raises
    ------
    RuntimeError
        - Error raised if the TuckerTensorTrains is internally inconsistent

    See Also
    --------
    TuckerTensorTrain
    t3_add
    t3_scale
    t3_sub

    Examples
    --------
    >>> import numpy as np
    >>> from t3tools import *
    >>> x = t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> neg_x = t3_neg(x)
    >>> print(np.linalg.norm(t3_to_dense(x) + t3_to_dense(neg_x)))
    0.0
    """
    return t3_scale(x, -1.0)


def t3_sub(
        x: TuckerTensorTrain,
        y: TuckerTensorTrain,
        use_jax: bool = False,
) -> TuckerTensorTrain:
    """Subtract TuckerTensorTrains x, y with the same shape, yielding a TuckerTensorTrain z=x-y with summed ranks.

    Parameters
    ----------
    x: TuckerTensorTrain
        First summand. structure=((N1,...,Nd), (n1,...,nd), (1, r1,...,r(d-1),1))
    y: TuckerTensorTrain
        Second summand. structure=((N1,...,Nd), (m1,...,md), (1, q1,...,q(d-1),1))
    use_jax: bool
        If True, returned TuckerTensorTrain cores are jnp.ndaray. Otherwise, np.ndarray. Default: False

    Returns
    -------
    z: TuckerTensorTrain
        Difference of Tucker tensor trains, z=x-y. structure=((N1,...,Nd), (n1+m1,...,nd+md), (1, r1+q1,...,r(d-1)+q(d-1),1))

    Raises
    ------
    RuntimeError
        - Error raised if either of the TuckerTensorTrains are internally inconsistent
        - Error raised if the TuckerTensorTrains have different shapes.

    See Also
    --------
    TuckerTensorTrain
    t3_shape
    t3_add
    t3_scale
    t3_neg

    Examples
    --------
    >>> import numpy as np
    >>> from t3tools import *
    >>> x = t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> y = t3_corewise_randn(((14,15,16), (3,7,2), (1,5,6,1)))
    >>> z = t3_sub(x, y)
    >>> print(t3_structure(z))
    ((14, 15, 16), (7, 12, 8), (1, 8, 8, 1))
    >>> print(np.linalg.norm(t3_to_dense(x) - t3_to_dense(y) - t3_to_dense(z)))
    3.5875705233607603e-13
    """
    return t3_add(x, t3_neg(y), use_jax=use_jax)


def t3_dot_t3(
        x: TuckerTensorTrain,
        y: TuckerTensorTrain,
        use_jax: bool = False,
):
    """Compute Hilbert-Schmidt (dot) product of two TuckerTensorTrains x, y with the same shape, (x, y)_HS.

    Parameters
    ----------
    x: TuckerTensorTrain
        First Tucker tensor train. shape=(N1,...,Nd)
    y: TuckerTensorTrain
        Second Tucker tensor train. shape=(N1,...,Nd)
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    scalar
        Hilbert-Schmidt (dot) product of Tucker tensor trains, (x, y)_HS.

    Raises
    ------
    RuntimeError
        - Error raised if either of the TuckerTensorTrains are internally inconsistent
        - Error raised if the TuckerTensorTrains have different shapes.

    See Also
    --------
    TuckerTensorTrain
    t3_shape
    t3_add
    t3_scale

    Notes
    -----
    Algorithm contracts the TuckerTensorTrains in a zippering fashion from left to right.

    Examples
    --------
    >>> import numpy as np
    >>> from t3tools import *
    >>> x = t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> y = t3_corewise_randn(((14,15,16), (3,7,2), (1,5,6,1)))
    >>> x_dot_y = t3_dot_t3(x, y)
    >>> x_dot_y2 = np.sum(t3_to_dense(x) * t3_to_dense(y))
    >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
    8.731149137020111e-11
    """
    t3_check(x)
    t3_check(y)

    xnp = jnp if use_jax else np

    x_shape = t3_structure(x)[0]
    y_shape = t3_structure(y)[0]
    if x_shape != y_shape:
        raise RuntimeError(
            'Attempted to dot TuckerTensorTrains (x,y)_HS with inconsistent shapes.'
            + str(x_shape) + ' = x_shape != y_shape = ' + str(y_shape)
        )

    basis_cores_x, tt_cores_x = x
    basis_cores_y, tt_cores_y = y

    M_sp = xnp.ones((1,1))
    for Bx_ai, Gx_sat, By_bi, Gy_pbq in zip(basis_cores_x, tt_cores_x, basis_cores_y, tt_cores_y):
        tmp_ab = xnp.einsum('ai,bi->ab', Bx_ai, By_bi)
        tmp_sbt = xnp.einsum('sat,ab->sbt', Gx_sat, tmp_ab)
        tmp_pbt = xnp.einsum('sp,sbt->pbt', M_sp, tmp_sbt)
        tmp_tq = xnp.einsum('pbt,pbq->tq', tmp_pbt, Gy_pbq)
        M_sp = tmp_tq

    return M_sp[0,0]


def t3_norm(
        x: TuckerTensorTrain,
        use_orthogonalization: bool = True,
        use_jax: bool = False,
):
    """Compute Hilbert-Schmidt (dot) product of two TuckerTensorTrains x, y with the same shape, (x, y)_HS.

    Parameters
    ----------
    x: TuckerTensorTrain
        First Tucker tensor train. shape=(N1,...,Nd)
    use_orthogonalization: bool
        If True, use orthogonalization-based algorithm (more stable).
        If False, use zippering algorithm (faster, easier to differentiate).
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    scalar
        Hilbert-Schmidt (Frobenius) norm of Tucker tensor trains, ||x||_HS

    Raises
    ------
    RuntimeError
        - Error raised if the TuckerTensorTrain is internally inconsistent

    See Also
    --------
    TuckerTensorTrain
    t3_shape
    t3_dot_t3

    Examples
    --------
    >>> import numpy as np
    >>> from t3tools import *
    >>> x = t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> norm_x = t3_norm(x)
    >>> print(np.abs(norm_x - np.linalg.norm(t3_to_dense(x))))
    1.3642420526593924e-12
    """
    t3_check(x)
    xnp = jnp if use_jax else np

    if use_orthogonalization:
        shape, _, _ = t3_structure(x)
        last_ind = len(shape)-1
        x2 = orthogonalize_relative_to_ith_basis_core(last_ind, x, use_jax=use_jax)
        basis_cores, tt_cores = x2
        return xnp.linalg.norm(basis_cores[-1])
    else:
        return xnp.sqrt(t3_dot_t3(x, x, use_jax=use_jax))

