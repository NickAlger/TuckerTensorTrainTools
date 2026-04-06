import numpy as np
import jax
import jax.numpy as jnp
import typing as typ

from tttt.tensor.tt_basic_operations import *
from tttt.tensor.t3_basic_operations import *
from tttt.tensor.t3_tangent_space import *
from tttt.tensor.actions import *

__all__ = [
    # Actions of a Tucker tensor train
    't3_actions',
    't3_tangent_actions',
    't3_tangent_actions_transpose',
    # Actions of a tangent vector
    't3_compute_mus',
    't3_compute_nus',
    't3_compute_sigmas',
    't3_compute_taus',
    't3_assemble_tangent_actions',
    # Transpose of tangent vector to actions map
    't3_compute_tau_tildes',
    't3_compute_sigma_tildes',
    't3_assemble_core_perturbations',
]

#

def t3_actions(
        x: TuckerTensorTrain,
        input_vectors:  typ.Sequence[jnp.ndarray], # inputs, len=k, elm_shape=(Ni,)
) -> typ.Tuple[jnp.ndarray,...]: # len=k, elm_shape=(Ni,)
    '''Actions of a Tucker tensor train'''
    basis_cores, tt_cores = x
    reduced_inputs = [jnp.einsum('io,o->i', B, v) for B, v in zip(basis_cores, input_vectors)]
    reduced_actions = tt_actions(tt_cores, reduced_inputs)
    actions = [jnp.einsum('i,io->o', a, B) for B, a in zip(basis_cores, reduced_actions)]
    return tuple(actions)


def t3_compute_mus(
        TS,
        reduced_xx,
):
    '''Left-to-right pushthroughs used to compute actions of a Tucker tensor train'''
    mus = [jnp.ones(1)]
    for ii in range(TS.num_cores-1):
        P = TS.left_orthogonal_tt_cores[ii]
        x = reduced_xx[ii]

        mu = mus[-1]
        mu_next = mu  @ jnp.einsum('iaj,a->ij', P, x) # jnp.einsum('i,ij->j', mu,     jnp.einsum('iaj,a->ij', P, x))
        mus.append(mu_next)
    return tuple(mus)


def t3_compute_nus(
        TS,
        reduced_xx,
):
    '''Right-to-left pushthroughs used to compute actions of a Tucker tensor train'''
    nus_reversed = [jnp.ones(1)]
    for ii in range(TS.num_cores-1, 0, -1):
        Q = TS.right_orthogonal_tt_cores[ii]
        x = reduced_xx[ii]

        nu = nus_reversed[-1]

        nu_prev     = jnp.einsum('iaj,a->ij', Q, x) @ nu # jnp.einsum('ij,j->i', jnp.einsum('iaj,a->ij', Q, x),  nu)
        nus_reversed.append(nu_prev)
    nus = nus_reversed[::-1]
    return tuple(nus)


def t3_compute_sigmas(
        tt_variations,
        TS,
        reduced_xx,
        reduced_dxx,
        mus,
):
    '''Left-to-right pushthrough partial sums used to compute actions of a Tucker tensor train tangent vector'''
    sigmas = [jnp.zeros(1)]
    for ii in range(TS.num_cores-1):
        Q = TS.right_orthogonal_tt_cores[ii]
        R = TS.up_orthogonal_tt_cores[ii]
        dU = tt_variations[ii]
        x = reduced_xx[ii]
        dx = reduced_dxx[ii]

        mu = mus[ii]
        sigma = sigmas[-1]

        sigma_next_t1   = sigma @ jnp.einsum('iaj,a->ij', Q, x) # jnp.einsum('i,ij->j', sigma,  jnp.einsum('iaj,a->ij', Q, x))
        sigma_next_t2   = mu    @ jnp.einsum('iaj,a->ij', dU, x) # jnp.einsum('i,ij->j', mu,     jnp.einsum('iaj,a->ij', dU, x))
        sigma_next_t3   = mu    @ jnp.einsum('iaj,a->ij', R, dx) # jnp.einsum('i,ij->j', mu,     jnp.einsum('iaj,a->ij', R, dx))

        sigma_next = sigma_next_t1 + sigma_next_t2 + sigma_next_t3
        sigmas.append(sigma_next)
    return tuple(sigmas)


def t3_compute_taus(
        tt_variations,
        TS,
        reduced_xx,
        reduced_dxx,
        nus,
):
    '''Right-to-left pushthrough partial sums used to compute actions of a Tucker tensor train tangent vector'''
    taus_reversed = [jnp.zeros(1)]
    for ii in range(TS.num_cores-1, 0, -1):
        P = TS.left_orthogonal_tt_cores[ii]
        R = TS.up_orthogonal_tt_cores[ii]
        dU = tt_variations[ii]
        x = reduced_xx[ii]
        dx = reduced_dxx[ii]

        nu = nus[ii]
        tau = taus_reversed[-1]

        tau_prev_t1 = jnp.einsum('iaj,a->ij', P, x) @ tau # jnp.einsum('ij,j->i', jnp.einsum('iaj,a->ij', P, x),  tau)
        tau_prev_t2 = jnp.einsum('iaj,a->ij', dU, x) @ nu # jnp.einsum('ij,j->i', jnp.einsum('iaj,a->ij', dU, x), nu)
        tau_prev_t3 = jnp.einsum('iaj,a->ij', R, dx) @ nu # jnp.einsum('ij,j->i', jnp.einsum('iaj,a->ij', R, dx), nu)

        tau_prev = tau_prev_t1 + tau_prev_t2 + tau_prev_t3
        taus_reversed.append(tau_prev)
    taus = taus_reversed[::-1]
    return tuple(taus)


def t3_assemble_tangent_actions(
        TS,
        tt_variations,
        basis_variations,
        mus,
        nus,
        sigmas,
        taus,
):
    '''Form actions of a Tucker tensor train tangent vector from already computed mus, nus, sigmas, taus'''
    actions = []
    for ii in range(TS.num_cores):
        P = TS.left_orthogonal_tt_cores[ii]
        Q = TS.right_orthogonal_tt_cores[ii]
        R = TS.up_orthogonal_tt_cores[ii]
        dU = tt_variations[ii]
        B = TS.orthogonal_basis_cores[ii]
        dB = basis_variations[ii]

        mu = mus[ii]
        nu = nus[ii]
        sigma = sigmas[ii]
        tau = taus[ii]

        # Actions given by following formula:
        # a(z) = [sigma, mu] [Q(B z),                 0] [nu]
        #                    [dU(B z) + R(dB z), P(B z)] [tau]

        reduced_action_t1 = sigma @ (Q @ nu) #jnp.einsum('i,iaj,j->a', sigma, Q, nu)
        reduced_action_t2 = mu @ (dU @ nu) #jnp.einsum('i,iaj,j->a', mu,    dU, nu)
        reduced_action_t3 = mu @ (R @ nu) #jnp.einsum('i,iaj,j->a', mu,    R, nu)
        reduced_action_t4 = mu @ (P @ tau) #jnp.einsum('i,iaj,j->a', mu,    P, tau)

        action = (reduced_action_t1 + reduced_action_t2 + reduced_action_t4) @ B + reduced_action_t3 @ dB
        actions.append(action)
    return tuple(actions)


def t3_tangent_actions(
        u:  T3Variations,
        TS: T3TangentSpace, # shape=(N1, N2, ..., Nk)
        input_vectors:  typ.Sequence[jnp.ndarray], # inputs, len=k, elm_shape=(Ni,)
) -> typ.Tuple[jnp.ndarray,...]: # len=k, elm_shape=(Ni,)
    '''Compute actions of a Tucker tensor train tangent vector.
        u(x,y,z,w) = G1(x) G2(y) G3(z) G4(w)
        where:
            G1(x) = [dU1(B x) + R1(dB x), P1(B x)]

            G2(y) = [Q2(B y),                   0]
                    [dU2(B y) + R2(dB y), P2(B y)]

            G3(z) = [Q3(B z),                   0]
                    [dU3(B z) + R3(dB z), P3(B z)]

            G4(w) = [Q4(B w) ]
                    [dU4(B w) + R4(dB w)]


    '''
    TS._check_consistency_with_other_ttt_cores(u)
    basis_variations, tt_variations = u

    reduced_xx  = [B  @ x for B,  x in zip(TS.orthogonal_basis_cores, input_vectors)]
    reduced_dxx = [dB @ x for dB, x in zip(basis_variations, input_vectors)]

    mus     = t3_compute_mus(TS, reduced_xx)
    sigmas  = t3_compute_sigmas(tt_variations, TS, reduced_xx, reduced_dxx, mus)
    nus     = t3_compute_nus(TS, reduced_xx)
    taus    = t3_compute_taus(tt_variations, TS, reduced_xx, reduced_dxx, nus)

    actions = t3_assemble_tangent_actions(TS, tt_variations, basis_variations, mus, nus, sigmas, taus)
    return actions


#

def t3_compute_tau_tildes(
        TS,
        reduced_xx,
        reduced_dyy,
        mus,
):
    '''Adjoints for right-to-left pushthrough partial sums (adjoints go the other way, left to right).
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.
    '''
    tau_tildes = [jnp.zeros(1)]
    for ii in range(TS.num_cores-1):
        P = TS.left_orthogonal_tt_cores[ii]
        x = reduced_xx[ii]
        dy = reduced_dyy[ii]

        mu = mus[ii]
        tau_tilde = tau_tildes[-1]

        tau_tilde_next_t1   = tau_tilde @ jnp.einsum('iaj,a->ij', P, x)
        tau_tilde_next_t2   = mu        @ jnp.einsum('iaj,a->ij', P, dy)

        tau_tilde_next = tau_tilde_next_t1 + tau_tilde_next_t2
        tau_tildes.append(tau_tilde_next)
    return tuple(tau_tildes)


def t3_compute_sigma_tildes(
        TS,
        reduced_xx,
        reduced_dyy,
        nus,
):
    '''Adjoints for left-to-right pushthrough partial sums (adjoints go the other way, right-to-left).
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.
    '''
    sigma_tildes_reversed = [jnp.zeros(1)]
    for ii in range(TS.num_cores-1, 0, -1):
        Q = TS.right_orthogonal_tt_cores[ii]
        x = reduced_xx[ii]
        dy = reduced_dyy[ii]

        nu = nus[ii]
        sigma_tilde = sigma_tildes_reversed[-1]

        sigma_tilde_prev_t1 = jnp.einsum('iaj,a->ij', Q, x)  @ sigma_tilde
        sigma_tilde_prev_t2 = jnp.einsum('iaj,a->ij', Q, dy) @ nu

        sigma_tilde_prev = sigma_tilde_prev_t1 + sigma_tilde_prev_t2
        sigma_tildes_reversed.append(sigma_tilde_prev)
    sigma_tildes = sigma_tildes_reversed[::-1]
    return tuple(sigma_tildes)


def t3_assemble_core_perturbations(
        TS,
        xx,
        dyy,
        reduced_xx,
        reduced_dyy,
        mus,
        nus,
        sigma_tildes,
        tau_tildes,
):
    '''Apply transpose of mapping from Tucker tensor train tangent vector to its actions,
    using already computed mus, nus, sigmas, taus.
    '''
    tt_core_perturbations = []
    for ii in range(TS.num_cores):
        mu = mus[ii]
        nu = nus[ii]
        sigma_tilde = sigma_tildes[ii]
        tau_tilde = tau_tildes[ii]
        x_hat = reduced_xx[ii]
        dy_hat = reduced_dyy[ii]

        dU_t1 = jnp.einsum('i,a,j->iaj', mu,        x_hat,  sigma_tilde)
        dU_t2 = jnp.einsum('i,a,j->iaj', tau_tilde, x_hat,  nu)
        dU_t3 = jnp.einsum('i,a,j->iaj', mu,        dy_hat, nu)

        dU = dU_t1 + dU_t2 + dU_t3
        tt_core_perturbations.append(dU)

    basis_core_perturbations = []
    for ii in range(TS.num_cores):
        R = TS.up_orthogonal_tt_cores[ii]
        mu = mus[ii]
        nu = nus[ii]
        sigma_tilde = sigma_tildes[ii]
        tau_tilde = tau_tildes[ii]
        x = xx[ii]
        dy = dyy[ii]

        dB_t1 = jnp.outer(jnp.einsum('i,iaj,j->a', mu,          R, sigma_tilde),    x)
        dB_t2 = jnp.outer(jnp.einsum('i,iaj,j->a', tau_tilde,   R, nu),             x)
        dB_t3 = jnp.outer(jnp.einsum('i,iaj,j->a', mu,          R, nu),             dy)

        dB = dB_t1 + dB_t2 + dB_t3
        basis_core_perturbations.append(dB)

    return tuple(tt_core_perturbations), tuple(basis_core_perturbations)


def t3_tangent_actions_transpose(
        action_perturbations: typ.Sequence[jnp.ndarray],  # inputs, len=k, elm_shape=(Ni,)
        TS: T3TangentSpace, # shape=(N1, N2, ..., Nk)
        input_vectors:  typ.Sequence[jnp.ndarray], # inputs, len=k, elm_shape=(Ni,)
) -> T3Variations:
    '''Transpose of mapping u -> ttt_tangent_actions(u, TS, input_vectors), where TS, input_vectors are fixed

    '''
    xx = input_vectors
    dyy = action_perturbations

    reduced_xx  = [B @ x for B,  x in zip(TS.orthogonal_basis_cores, xx)]
    reduced_dyy = [B @ dy for B, dy in zip(TS.orthogonal_basis_cores, dyy)]

    mus = t3_compute_mus(TS, reduced_xx)
    nus = t3_compute_nus(TS, reduced_xx)
    tau_tildes = t3_compute_tau_tildes(TS, reduced_xx, reduced_dyy, mus)
    sigma_tildes = t3_compute_sigma_tildes(TS, reduced_xx, reduced_dyy, nus)

    tt_core_perturbations, basis_core_perturbations = t3_assemble_core_perturbations(
        TS, xx, dyy, reduced_xx, reduced_dyy, mus, nus, sigma_tildes, tau_tildes,
    )

    return basis_core_perturbations, tt_core_perturbations



