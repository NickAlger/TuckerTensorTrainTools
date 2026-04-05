# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import typing as typ
from t3tools.tucker_tensor_train import *
from t3tools.t3_base_variation_format import *
from t3tools.t3_orthogonalization import *
from t3tools.t3_svd import *

try:
    import jax.numpy as jnp
except:
    print('jax import failed in tucker_tensor_train. Defaulting to numpy.')
    jnp = np

NDArray = typ.Union[np.ndarray, jnp.ndarray]


__all__ = [
    'T3Tangent',
    't3tangent_to_dense',
    't3tangent_to_t3',
    't3_orthogonal_gauge_projection',
    't3_oblique_gauge_projection',
    'project_t3_onto_tangent_space',
    't3_retract',
    't3tangent_zeros',
    't3tangent_randn',
    't3tangent_scale',
    't3tangent_add',
    't3tangent_neg',
    't3tangent_sub',
    't3tangent_dot_t3tangent',
    't3tangent_norm',
]


####################################################################
###########    Fixed rank TuckerTensorTrain manifold   #############
####################################################################

T3Tangent = typ.Tuple[
    T3Base, # Orthogonal representations of base point
    T3Variation, # Variations
]
"""Tuple representation of a Tucker tensor train tangent vector in terms of an orthogonal base and variations.

Implements the representation of a tangent vector as a sum of the following form::

          1 -- H0 -- R1 -- R2 -- R3 -- 1            1 -- L0 -- L1 -- L2 -- H3 -- 1
               |     |     |     |                       |     |     |     |
    v  =       U0    U1    U2    U3       +  ... +       U0    U1    U2    U3
               |     |     |     |                       |     |     |     |
    
          1 -- O0 -- R1 -- R2 -- R3 -- 1            1 -- L0 -- L1 -- L2 -- O3 -- 1
       +       |     |     |     |                       |     |     |     |
               V0    U1    U2    U3       + ... +        U0    U1    U2    V3
               |     |     |     |                       |     |     |     |

**Components**
    - T3Base: orthogonal representation of the base point where the tangent space is attached.
        - basis_cores       = (U0,...,Ud), orthogonal
        - left_tt_cores     = (L0,...Ld), left-orthogonal
        - right_tt_cores    = (R0,...,Rd), right-orthogonal
        - outer_tt_cores    = (O0,...,Od), outer-orthogonal
    - T3Variation: The variations defining the tangent vector w.r.t. the base point.
        - basis_variations  = (V0,...,Vd)
        - tt_variations     = (H0,...,Hd)

Examples
--------
>>> from numpy.random import randn
>>> from t3tools.t3_manifold import *
>>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
>>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
>>> p = (basis_cores, tt_cores)
>>> base, _ = t3_orthogonal_representations(p)
>>> other_basis_cores = (randn(7,14), randn(4,15), randn(8,16)) # same shape, different ranks
>>> other_tt_cores = (randn(1,7,5), randn(5,4,4), randn(4,8,1))
>>> x = (other_basis_cores, other_tt_cores)
>>> proj_x = project_t3_onto_tangent_space(x, base)
"""

def t3tangent_to_dense(
        x: T3Tangent,
        include_shift: bool = False, # False: V. True: P+V. P=base point, V=tangent vector
) -> NDArray:
    """Convert Tangent vector to Tucker tensor train manifold into dense tensor.

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> v = x[1]
    >>> v_dense = t3tangent_to_dense(x)
    >>> ((U0,U1,U2), (L0,L1,L2), (R0,R1,R2), (O0,O1,O2)) = base
    >>> ((V0,V1,V2), (H0,H1,H2)) = v
    >>> t1 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,U1,U2,H0,R1,R2)
    >>> t2 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,U1,U2,L0,H1,R2)
    >>> t3 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,U1,U2,L0,L1,H2)
    >>> t4 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', V0,U1,U2,O0,R1,R2)
    >>> t5 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,V1,U2,L0,O1,R2)
    >>> t6 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,U1,V2,L0,L1,O2)
    >>> v_dense2 = t1 + t2 + t3 + t4 + t5 + t6
    >>> print(np.linalg.norm(v_dense - v_dense2))
        1.2760924630140578e-14
    >>> p_plus_v_dense = t3tangent_to_dense(x, include_shift=True)
    >>> p_plus_v_dense2 =  t3_to_dense(p) + v_dense
    >>> print(np.linalg.norm(p_plus_v_dense - p_plus_v_dense2))
        1.2677102046134292e-12
    """
    t3_check_base_variation_fit(*x)

    base, variation = x
    num_cores = len(variation[0])
    basis_terms = [bv_to_t3(ii, False, base, variation) for ii in range(num_cores)]
    tt_terms    = [bv_to_t3(ii, True, base, variation) for ii in range(num_cores)]
    terms = basis_terms + tt_terms
    V = t3_to_dense(terms[0])
    for t in terms[1:]:
        V = V + t3_to_dense(t)

    if include_shift:
        basis_cores, left_tt_cores, _, _ = base
        P = t3_to_dense((basis_cores, left_tt_cores))
        X = P + V
    else:
        X = V

    return X


def t3tangent_to_t3(
        x: T3Tangent,
        include_shift: bool = False,  # False: v. True: p+v. p=base point, v=tangent vector
        use_jax: bool = False,
) -> TuckerTensorTrain:
    '''Rank 2r Tucker tensor train representation of tangent vector.

    Without shift, we use the formula::

        v(x,y,z,w) = ([dU1(B x) L1(B x)]) ([R2(B y)        0]) ([R3(B z)        0]) ([R4(B w) ])
                     (                  ) ([dU2(B y) L2(B y)]) ([dU3(B z) L3(B z)]) ([dU4(B w)])
                     (         +        ) (         +        ) (        +         ) (    +     )
                     ([O1(dB x)       0]) ([0              0]) ([0              0]) ([0       ])
                     (                  ) ([O2(dB y)       0]) ([O3(dB z)       0]) ([O4(dB w)])

    With shift is same as unshifted, except last core modified as follows::

        [R4(B w) ]                  [R4(B w)           ]
        [dU4(B w)]                  [L4(B w) + dU4(B w)]
            +             ->            +
        [0       ]                  [0                 ]
        [O4(dB w)]                  [O4(dB w)          ]

    Parameters
    ----------
    x: T3Tangent
        Tangent vector which will be converted to TuckerTensorTrain with doubled ranks.
    include_shift: bool
        If False, return tangent vector v only. If True, shift tangent vector so it is attached at the base point, p+v.
    use_jax: bool
        If True, returned TuckerTensorTrain cores are jnp.ndaray. Otherwise, np.ndarray. Default: False

    Returns
    -------
    TuckerTensorTrain
        Tucker tensor train representation of tangent vector, which has doubled ranks

    See Also
    --------
    T3Tangent

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> v_t3 = t3tangent_to_t3(x) # tangent vector only (attached at zero)
    >>> v_dense = t3_to_dense(v_t3)
    >>> v_dense2 = t3tangent_to_dense(x)
    >>> print(np.linalg.norm(v_dense - v_dense2))
        2.678565538404836e-15
    >>> p_plus_v_t3 = t3tangent_to_t3(x, include_shift=True) # shifted tangent vector (include attachment at base point)
    >>> p_plus_v_dense = t3_to_dense(p_plus_v_t3)
    >>> p_plus_v_dense2 = v_dense2 + t3_to_dense(p)
    >>> print(np.linalg.norm(p_plus_v_dense - p_plus_v_dense2))
        1.2102169224182523e-12
    '''
    xnp = jnp if use_jax else np

    t3_check_base_variation_fit(*x)
    base, vars = x
    basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    basis_vars, tt_vars = vars

    num_cores = len(basis_cores)

    x_basis_cores = []
    for B, dB in zip(basis_cores, basis_vars):
        B2 = xnp.concatenate([B, dB], axis=0)
        x_basis_cores.append(B2)

    x_tt_cores = []

    dU = tt_vars[0]
    O = outer_tt_cores[0]
    L = left_tt_cores[0]
    Z = xnp.zeros((O.shape[0], O.shape[1], L.shape[2]))
    G_top = xnp.concatenate([dU, L], axis=2)
    G_bot = xnp.concatenate([O, Z], axis=2)
    G = xnp.concatenate([G_top, G_bot], axis=1)
    x_tt_cores.append(G)

    for ii in range(1, num_cores-1):
        L = left_tt_cores[ii]
        R = right_tt_cores[ii]
        O = outer_tt_cores[ii]
        dU = tt_vars[ii]
        Z001 = xnp.zeros((R.shape[0], dU.shape[1], L.shape[2]))
        Z100 = xnp.zeros((R.shape[0], O.shape[1], R.shape[2]))
        Z101 = xnp.zeros((R.shape[0], O.shape[1], L.shape[2])) #Z001
        Z111 = xnp.zeros((L.shape[0], O.shape[1], L.shape[2])) #jnp.zeros(L.shape)
        G_top = xnp.concatenate([
            xnp.concatenate([R, Z001], axis=2),
            xnp.concatenate([dU, L], axis=2)
        ], axis=0)
        G_bot = xnp.concatenate([
            xnp.concatenate([Z100, Z101], axis=2),
            xnp.concatenate([O, Z111], axis=2)
        ], axis=0)
        G = xnp.concatenate([G_top, G_bot], axis=1)
        x_tt_cores.append(G)

    dU = tt_vars[-1]
    L = left_tt_cores[-1]
    R = right_tt_cores[-1]
    O = outer_tt_cores[-1]
    Z = xnp.zeros((R.shape[0], O.shape[1], R.shape[2]))
    if include_shift:
        G_top = xnp.concatenate([R, L + dU], axis=0)
    else:
        G_top = xnp.concatenate([R, dU], axis=0)
    G_bot = xnp.concatenate([Z, O], axis=0)
    G = xnp.concatenate([G_top, G_bot], axis=1)
    x_tt_cores.append(G)

    return tuple(x_basis_cores), tuple(x_tt_cores)


def t3_orthogonal_gauge_projection(
        x: T3Tangent,
        use_jax: bool = False,
) -> T3Tangent:
    """Makes tangent vector representation gauged via orthogonal projection. Changes tangent vector.

    Gauge condition:
        - All variation basis cores Vi are orthogonal to the corresponding base basis cores Ui:
            Ui @ Vi.T = 0    for    i=1,...,d
        - All but the last variation TT-cores H are left-perpendicular to the corresponding base left TT-cores L:
            einsum('iaj,iak->jk', Hi, Li) = 0    for    i=1,...,d-1

    Parameters
    ----------
    x: T3Tangent
        Tangent vector to project

    use_jax: bool
        If True, use jax operations, if False use numpy.

    Returns
    -------
    T3Tangent
        Projected tangent vector satisfying Gauge condition

    See Also
    --------
    T3Tangent
    t3_oblique_gauge_projection

    Example
    -------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> projected_x = t3_orthogonal_gauge_projection(x)
    >>> (U0,U1,U2), (L0,L1,L2), _, _ = base
    >>> ((V0,V1,V2), (H0,H1,H2)) = projected_x[1]
    >>> print(np.linalg.norm(V1 @ U1.T)) # Gauge condition for basis core 1
        3.512073125137391e-15
    >>>  print(np.linalg.norm(np.einsum('iaj,iak->jk', H1, L1))) # Gauge condition for TT-core 1
        1.5807940730805242e-15
    """
    xnp = jnp if use_jax else np

    t3_check_base_variation_fit(*x)
    base, vars = x
    basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    basis_vars, tt_vars = vars

    new_tt_variations = []
    for dV, P in zip(tt_vars[:-1], left_tt_cores[:-1]):
        dV2 = dV - xnp.einsum('iaj,jk->iak', P, xnp.einsum('iaj,iak->jk', P, dV))
        new_tt_variations.append(dV2)
    new_tt_variations.append(tt_vars[-1])

    new_basis_variations = []
    for dB, B in zip(basis_vars, basis_cores):
        dB2 = dB - (dB @ B.T) @ B
        new_basis_variations.append(dB2)

    new_vars = (tuple(new_basis_variations), tuple(new_tt_variations))
    new_x = (base, new_vars)
    return new_x


def t3_oblique_gauge_projection(
        x: T3Tangent,
        use_jax: bool = False,
) -> T3Tangent:
    """Makes variations left-perpendicular while preserving tangent vector.

    Method:
        1) Make basis variations left-perpendicular by pushing remainder onto tt variations
        2) Make tt variations left-perpendicular by standard sweeping tt method

    Parameters
    ----------
    x: T3Tangent
        Tangent vector to project

    use_jax: bool
        If True, use jax operations, if False use numpy.

    Returns
    -------
    T3Tangent
        Projected tangent vector satisfying Gauge condition

    See Also
    --------
    T3Tangent
    t3_orthogonal_gauge_projection

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> projected_x = t3_oblique_gauge_projection(x)
    >>> x_dense = t3tangent_to_dense(x)
    >>> proj_x_dense = t3tangent_to_dense(projected_x)
    >>> print(np.linalg.norm(x_dense - proj_x_dense)) # Zero since projection preserves represented tangent vector
        3.4398319441148304e-15
    >>> (U0,U1,U2), (L0,L1,L2), _, _ = base
    >>> ((V0,V1,V2), (H0,H1,H2)) = projected_x[1]
    >>> print(np.linalg.norm(V1 @ U1.T)) # Gauge condition for basis core 1
        2.931519226677228e-15
    >>>  print(np.linalg.norm(np.einsum('iaj,iak->jk', H1, L1))) # Gauge condition for TT-core 1
        6.99005312491287e-16
    """
    xnp = jnp if use_jax else np

    t3_check_base_variation_fit(*x)
    base, vars = x
    basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    basis_vars, tt_vars = vars
    num_cores = len(basis_cores)

    tt_vars = list(tt_vars)
    basis_vars = list(basis_vars)

    # Make basis variations left-perpendicular
    for ii in range(num_cores):
        B_io = basis_cores[ii]
        dB_jo = basis_vars[ii]
        R_aib = outer_tt_cores[ii]
        dG_ajb = tt_vars[ii]

        X_ji = dB_jo @ B_io.T
        dB_parallel_jo = X_ji @ B_io
        dB2_jo = dB_jo - dB_parallel_jo # dB_perp
        # dG2_ajb = dG_ajb + xnp.einsum('aib,ji->ajb', R_aib, X_ji)
        dG2_ajb = dG_ajb + xnp.einsum('aib,ij->ajb', R_aib, X_ji) # <-- Why is this correct?

        tt_vars[ii] = dG2_ajb
        basis_vars[ii] = dB2_jo

    # Make tt cores left-perpendicular
    for ii in range(num_cores-1):
        dV1 = tt_vars[ii]
        dV2 = tt_vars[ii+1]

        P1 = left_tt_cores[ii]
        Q2 = right_tt_cores[ii+1]
        X = xnp.einsum('iaj,iak->jk', P1, dV1)
        new_dV1 = dV1 - xnp.einsum('iaj,jk->iak', P1, X)
        new_dV2 = dV2 + xnp.einsum('jk,kbl->jbl', X, Q2)

        tt_vars[ii] = new_dV1
        tt_vars[ii+1] = new_dV2

    new_vars = tuple(basis_vars), tuple(tt_vars)
    new_x = (base, new_vars)
    return new_x


def tt_reverse(cores):
    return tuple([G.swapaxes(0, 2) for G in cores[::-1]])


def tt_zipper_left_to_right(
        coresA: typ.Sequence[NDArray],
        coresB: typ.Sequence[NDArray],
        use_jax: bool = False,
) -> typ.Tuple[NDArray, ...]:  # zipper_matrices. len=num_cores+1
    xnp = jnp if use_jax else np

    zipper_matrices = [xnp.array([[1.0]])]
    for GA, GB in zip(coresA, coresB):
        Z_prev = zipper_matrices[-1]
        Z = xnp.einsum('ij,iak,jal->kl', Z_prev, GA, GB)
        zipper_matrices.append(Z)
    return tuple(zipper_matrices)


def tt_zipper_right_to_left(
        coresA: typ.Sequence[NDArray],
        coresB: typ.Sequence[NDArray],
        use_jax: bool = False,
) -> typ.Tuple[NDArray, ...]:  # zipper_matrices. len=num_cores+1
    return tt_zipper_left_to_right(tt_reverse(coresA), tt_reverse(coresB), use_jax=use_jax)[::-1]


def project_t3_onto_tangent_space(
        x: TuckerTensorTrain, # Tucker tensor train to be projected
        orthogonal_base: T3Base, # Orthogonal representations of base point
        use_jax: bool = False,
) -> T3Tangent:
    """Projects TuckerTensorTrain onto tangent space to the manifold of fixed rank TuckerTensorTrains.

    Parameters
    ----------
    x: T3Tangent
        Tangent vector to project
    orthogonal_base: T3Base
        Orthogonal representations of base point on manifold where tangent space is attached
    use_jax: bool
        If True, use jax operations, if False use numpy.

    Returns
    -------
    T3Tangent
        Projection of x onto the tangent space. Satisfies the gauge condition.

    See Also
    --------
    T3Tangent
    t3_oblique_gauge_projection
    t3_orthogonal_gauge_projection

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, _ = t3_orthogonal_representations(p)
    >>> other_basis_cores = (randn(7,14), randn(4,15), randn(8,16)) # same shape, different ranks
    >>> other_tt_cores = (randn(1,7,5), randn(5,4,4), randn(4,8,1))
    >>> x = (other_basis_cores, other_tt_cores)
    >>> proj_x = project_t3_onto_tangent_space(x, base)
    >>> P = t3_to_dense(p)
    >>> X = t3_to_dense(x)
    >>> proj_X = t3tangent_to_dense(proj_x)
    >>> print(np.sum((X - proj_X) * (proj_X - P)) / np.sum(X)) # Check that x was projected orthogonally
        3.351282624686308e-11
    """
    xnp = jnp if use_jax else np

    t3_check(x)
    t3_check_base(orthogonal_base)

    basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = orthogonal_base
    other_basis_cores, other_tt_cores = x

    base_shape = tuple([B.shape[1] for B in basis_cores])
    other_shape = tuple([B.shape[1] for B in other_basis_cores])
    if base_shape != other_shape:
        raise RuntimeError(
            'Attempted to retract TuckerTensorTrain with wrong shape onto tangent space.\n'
            + str(base_shape) + ' = base_shape != other_shape = ' + str(other_shape)
        )

    other_tt_cores2 = []
    for G_other, B_other, B in zip(other_tt_cores, other_basis_cores, basis_cores):
        G_other2 = xnp.einsum('aib,ix->axb', G_other, B_other @ B.T)
        other_tt_cores2.append(G_other2)

    zipper_left2right = tt_zipper_left_to_right(other_tt_cores2, left_tt_cores, use_jax=use_jax)[:-1]
    zipper_right2left = tt_zipper_right_to_left(other_tt_cores2, right_tt_cores, use_jax=use_jax)[1:]

    ungauged_tt_variations = []
    ungauged_basis_variations = []
    for ZL_ax, ZR_by, G_aib, B_io, R0_xjy, B0_jo in zip(
            zipper_left2right, zipper_right2left,
            other_tt_cores, other_basis_cores,
            outer_tt_cores, basis_cores,
    ):
        X_xiy = xnp.einsum('ax,aib,by->xiy', ZL_ax, G_aib, ZR_by)
        dG_xjy = xnp.einsum('xiy,ij->xjy', X_xiy, B_io @ B0_jo.T)
        M_ij = xnp.einsum('xiy,xjy->ij', X_xiy, R0_xjy)
        dB_jo = xnp.einsum('ij,io->jo', M_ij, B_io)

        ungauged_tt_variations.append(dG_xjy)
        ungauged_basis_variations.append(dB_jo)

    ungauged_u = (orthogonal_base, (ungauged_basis_variations, ungauged_tt_variations))
    gauged_u = t3_orthogonal_gauge_projection(ungauged_u)
    return gauged_u


def t3_retract(
        x: T3Tangent,
        use_jax: bool = False,
) -> TuckerTensorTrain: # retracted Tucker tensor train
    """Retract Tucker tensor train tangent vector to manifold.

    Parameters
    ----------
    x: T3Tangent
        Tangent vector to project
    orthogonal_base: T3Base
        Orthogonal representations of base point on manifold where tangent space is attached
    use_jax: bool
        If True, use jax operations, if False use numpy.

    Returns
    -------
    T3Tangent
        Projection of x onto the tangent space. Satisfies the gauge condition.

    See Also
    --------
    T3Tangent
    t3_svd

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> ret_x = t3_retract(x)
    >>> print(np.linalg.norm(t3_to_dense(ret_x) - t3tangent_to_dense(x, include_shift=True)))
        0.12395762641282583
    >>> x2 = t3tangent_scale(x, 1e-3) # make the tangent vector shorter for smaller retraction
    >>> ret_x2 = t3_retract(x2)
    >>> print(np.linalg.norm(t3_to_dense(ret_x2) - t3tangent_to_dense(x2, include_shift=True)))
        1.256998583616651e-07
    """
    t3_check_base_variation_fit(*x)

    basis_cores, left_tt_cores, _, _ = x[0]
    _, base_tucker_ranks, base_tt_ranks = t3_structure((basis_cores, left_tt_cores))

    x_t3 = t3tangent_to_t3(x, include_shift=True)
    retracted_x_t3, _, _ = t3_svd(
        x_t3,
        max_tt_ranks = base_tt_ranks,
        max_tucker_ranks = base_tucker_ranks,
        use_jax=use_jax,
    )
    return retracted_x_t3


####################################################################
############    Construct special kinds of T3Tangent   #############
####################################################################

def t3tangent_zeros(
        orthogonal_base: T3Base, # orthogonal base
        use_jax: bool = False,
) -> T3Tangent:
    """Construct the zero vector in a Tucker tensor train tangent space.

    Parameters
    ----------
    orthogonal_base: T3Base
        Orthogonal representations of base point on manifold where tangent space is attached
    use_jax: bool
        If True, return jax arrays, if False return numpy.

    Returns
    -------
    T3Tangent
        Zero vector in the tangent space

    See Also
    --------
    T3Tangent
    t3tangent_randn

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> z = t3tangent_zeros(base)
    >>> print(np.linalg.norm(t3tangent_to_dense(z)))
        0.0
    """
    xnp = jnp if use_jax else np

    t3_check_base(orthogonal_base)

    var_basis_shapes, var_tt_shapes = t3_base_hole_shapes(orthogonal_base)

    basis_vars = tuple([xnp.zeros(s) for s in var_basis_shapes])
    tt_vars = tuple([xnp.zeros(s) for s in var_tt_shapes])

    zero = (orthogonal_base, (basis_vars, tt_vars))
    return zero


def t3tangent_randn(
        orthogonal_base: T3Base, # orthogonal base
        use_jax: bool = False,
) -> T3Tangent:
    """Draw a random T3Tangent from a uniform normal disribution on the tangent space.

    Parameters
    ----------
    orthogonal_base: T3Base
        Orthogonal representations of base point on manifold where tangent space is attached

    Returns
    -------
    T3Tangent
        Random tangent vector. gauged.
    use_jax: bool
        If True, return jax arrays, if False return numpy. Should update this to use pure jax, rather than converting numpy->jax.

    See Also
    --------
    T3Tangent
    t3tangent_zeros

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    """
    t3_check_base(orthogonal_base)

    var_basis_shapes, var_tt_shapes = t3_base_hole_shapes(orthogonal_base)

    if use_jax:
        _randn = lambda x: jnp.array(np.random.randn(x))
    else:
        _randn = np.random.randn

    basis_vars0 = tuple([_randn(*s) for s in var_basis_shapes])
    tt_vars0 = tuple([_randn(*s) for s in var_tt_shapes])

    x0 = (orthogonal_base, (basis_vars0, tt_vars0))
    x = t3_orthogonal_gauge_projection(x0)
    return x


####################################################################
##################    T3Tangent linear algebra   ###################
####################################################################

def t3tangent_scale(
        x: T3Tangent,
        s, # scalar
) -> T3Tangent:
    """Scale T3Tangent.

    Parameters
    ----------
    x: T3Tangent
        Tangent vector to scale
    s: scalar
        Scaling factor

    Returns
    -------
    T3Tangent
        Scaled T3Tangent, s*x.

    See Also
    --------
    T3Tangent
    t3tangent_add
    t3tangent_sub
    t3tangent_neg
    t3tangent_dot_t3tangent
    t3tangent_norm

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> s = randn()
    >>> sx = t3tangent_scale(x, s)
    >>> dense_sx = t3tangent_to_dense(sx)
    >>> dense_x = t3tangent_to_dense(x)
    >>> print(np.linalg.norm(dense_sx - s*dense_x))
        2.370617783938243e-15
    """
    t3_check_base_variation_fit(*x)

    base, vars = x
    basis_vars, tt_vars = list(vars)
    new_basis_vars = tuple([s*B for B in basis_vars])
    new_tt_vars = tuple([s*G for G in tt_vars])

    new_x = (base, (new_basis_vars, new_tt_vars))
    return new_x


def t3tangent_add(
        x: T3Tangent,
        y: T3Tangent,
) -> T3Tangent:
    """Add T3Tangents, (x, y) -> x+y.

    Parameters
    ----------
    x: T3Tangent
        First tangent vector summand
    y: T3Tangent
        Second tangent vector summand

    Returns
    -------
    T3Tangent
        Sum of tangent vectors, x+y

    Raises
    ------
    RuntimeError
        Error raised if either T3Tangent is inconsistent, or if x and y have different bases.

    See Also
    --------
    T3Tangent
    t3tangent_scale
    t3tangent_sub
    t3tangent_neg
    t3tangent_dot_t3tangent
    t3tangent_norm

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> y = t3tangent_randn(base)
    >>> x_plus_y = t3tangent_add(x, y)
    >>> x_dense = t3tangent_to_dense(x)
    >>> y_dense = t3tangent_to_dense(y)
    >>> x_plus_y_dense = t3tangent_to_dense(x_plus_y)
    >>> print(np.linalg.norm(x_dense + y_dense - x_plus_y_dense))
        4.667707616068206e-15
    """
    t3_check_base_variation_fit(*x)
    t3_check_base_variation_fit(*y)

    x_base = x[0]
    y_base = y[0]
    if x_base != y_base:
        raise RuntimeError(
            'Attempted to add T3Tangent vectors with different bases.'
        )

    x_basis_vars, x_tt_vars = x[1]
    y_basis_vars, y_tt_vars = y[1]

    x_plus_y_basis_vars = tuple([Bx + By for Bx, By in zip(x_basis_vars, y_basis_vars)])
    x_plus_y_tt_vars = tuple([Gx + Gy for Gx, Gy in zip(x_tt_vars, y_tt_vars)])

    x_plus_y = (x_base, (x_plus_y_basis_vars, x_plus_y_tt_vars))
    return x_plus_y


def t3tangent_neg(
        x: T3Tangent,
) -> T3Tangent:
    """Negative of T3Tangent, x -> -x.

    Parameters
    ----------
    x: T3Tangent
        input tangent vector

    Returns
    -------
    T3Tangent
        Negative of x, i.e., -x.

    See Also
    --------
    T3Tangent
    t3tangent_add
    t3tangent_sub
    t3tangent_scale
    t3tangent_dot_t3tangent
    t3tangent_norm

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> neg_x = t3tangent_neg(x)
    >>> dense_neg_x = t3tangent_to_dense(neg_x)
    >>> dense_x = t3tangent_to_dense(x)
    >>> print(np.linalg.norm(dense_neg_x + dense_x))
        0.0
    """
    return t3tangent_scale(x, -1.0)


def t3tangent_sub(
        x: T3Tangent,
        y: T3Tangent,
) -> T3Tangent:
    """Subtract T3Tangents, (x,y) -> x-y.

    Parameters
    ----------
    x: T3Tangent
        First tangent vector summand
    y: T3Tangent
        Second tangent vector summand

    Returns
    -------
    T3Tangent
        Difference of tangent vectors, x-y

    Raises
    ------
    RuntimeError
        Error raised if either T3Tangent is inconsistent, or if x and y have different bases.

    See Also
    --------
    T3Tangent
    t3tangent_scale
    t3tangent_add
    t3tangent_neg
    t3tangent_dot_t3tangent
    t3tangent_norm

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> y = t3tangent_randn(base)
    >>> x_minus_y = t3tangent_sub(x, y)
    >>> x_dense = t3tangent_to_dense(x)
    >>> y_dense = t3tangent_to_dense(y)
    >>> x_minus_y_dense = t3tangent_to_dense(x_minus_y)
    >>> print(np.linalg.norm(x_dense - y_dense - x_minus_y_dense))
        4.714955371344249e-15
    """
    return t3tangent_add(x, t3tangent_scale(y, -1.0))


def t3tangent_dot_t3tangent(
        gauged_x: T3Tangent,
        gauged_y: T3Tangent,
        use_jax: bool = False,
):
    """Compute Hilbert-Schmidt inner product (dot product in ambient space) of T3Tangents, (x,y)_HS

    Parameters
    ----------
    x: T3Tangent
        First tangent vector. Must be gauged! If not, use t3_oblique_gauge_projection(x)
    y: T3Tangent
        Second tangent vector. Must be gauged! If not, use t3_oblique_gauge_projection(y)

    Returns
    -------
    Scalar
        Inner product of tangent vectors, (x,y)_HS

    Raises
    ------
    RuntimeError
        Error raised if either T3Tangent is inconsistent, or if x and y have different bases.

    See Also
    --------
    T3Tangent
    t3_oblique_gauge_projection
    t3_orthogonal_gauge_projection
    t3tangent_scale
    t3tangent_add
    t3tangent_neg
    t3tangent_sub
    t3tangent_norm

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> y = t3tangent_randn(base)
    >>> x_dot_y = t3tangent_dot_t3tangent(x, y)
    >>> x_dense = t3tangent_to_dense(x)
    >>> y_dense = t3tangent_to_dense(y)
    >>> x_dot_y2 = np.sum(x_dense * y_dense)
    >>> print(np.abs(x_dot_y - x_dot_y2))
        1.2434497875801753e-14
        
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x_basis_vars = tuple([randn(*B.shape) for B in vars0[0]])
    >>> x_tt_vars = tuple([randn(*G.shape) for G in vars0[1]])
    >>> x = (base, (x_basis_vars, x_tt_vars))
    >>> y_basis_vars = tuple([randn(*B.shape) for B in vars0[0]])
    >>> y_tt_vars = tuple([randn(*G.shape) for G in vars0[1]])
    >>> y = (base, (y_basis_vars, y_tt_vars))
    >>> bad_x_dot_y = t3tangent_dot_t3tangent(x, y) # x and y are ungauged, so this will not give the right answer
    >>> x_dense = t3tangent_to_dense(x)
    >>> y_dense = t3tangent_to_dense(y)
    >>> x_dot_y2 = np.sum(x_dense * y_dense)
    >>> print(np.abs(bad_x_dot_y - x_dot_y2))
        5.609998509008447
    >>> x_gauged = t3_oblique_gauge_projection(x) # make them gauged and try again
    >>> y_gauged = t3_oblique_gauge_projection(y)
    >>> x_dot_y = t3tangent_dot_t3tangent(x_gauged, y_gauged)
    >>> print(np.abs(x_dot_y - x_dot_y2))
        1.5987211554602254e-14
    """
    xnp = jnp if use_jax else np

    t3_check_base_variation_fit(*gauged_x)
    t3_check_base_variation_fit(*gauged_y)

    x_base = gauged_x[0]
    y_base = gauged_y[0]
    if x_base != y_base:
        raise RuntimeError(
            'Attempted to dot T3Tangent vectors with different bases.'
        )

    x_basis_vars, x_tt_vars = gauged_x[1]
    y_basis_vars, y_tt_vars = gauged_y[1]

    t1 = xnp.sum([np.sum(Bx*By) for Bx, By in zip(x_basis_vars, y_basis_vars)])
    t2 = xnp.sum([np.sum(Gx*Gy) for Gx, Gy in zip(x_tt_vars, y_tt_vars)])

    return t1+t2


def t3tangent_norm(
        gauged_x: T3Tangent,
        use_jax: bool = False,
):
    """Compute Hilbert-Schmidt (Frobenius) norm (in ambient space) of T3Tangent, ||x||_HS.

    Parameters
    ----------
    x: T3Tangent
        Tangent vector to compute the norm of. Must be gauged! If not, use t3_oblique_gauge_projection(x)

    Returns
    -------
    Scalar
        Norm of tangent vector,||x||_HS.

    Raises
    ------
    RuntimeError
        Error raised if x is inconsistent.

    See Also
    --------
    T3Tangent
    
    t3tangent_scale
    t3tangent_add
    t3tangent_neg
    t3tangent_sub
    t3tangent_dot_t3tangent

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> norm_x = t3tangent_norm(x)
    >>> x_dense = t3tangent_to_dense(x)
    >>> norm_x2 = np.linalg.norm(x_dense)
    >>> print(np.abs(norm_x - norm_x2))
        3.552713678800501e-15
    """
    xnp = jnp if use_jax else np
    return xnp.sqrt(t3tangent_dot_t3tangent(gauged_x, gauged_x, use_jax=use_jax))


