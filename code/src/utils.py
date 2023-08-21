import itertools
from typing import NamedTuple, Callable, Any, Union
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import custom_vjp, vjp
from jax.custom_derivatives import closure_convert
from jax.flatten_util import ravel_pytree
from jax.lax import while_loop


class MVNStandard(NamedTuple):
    mean: Any
    cov: Any


class MVNSqrt(NamedTuple):
    mean: Any
    chol: Any


class FunctionalModel(NamedTuple):
    function: Callable
    mvn: Union[MVNSqrt, MVNStandard]


class ConditionalMomentsModel(NamedTuple):
    conditional_mean: Callable
    conditional_covariance_or_cholesky: Callable

def are_inputs_compatible(*y):
    a, b = itertools.tee(map(type, y))
    _ = next(b, None)
    ok = sum(map(lambda u: u[0] == u[1], zip(a, b)))
    if not ok:
        raise TypeError(f"All inputs should have the same type. {y} was given")

def tria(A):
    return qr(A.T).T

def none_or_shift(x, shift):
    if x is None:
        return None
    if shift > 0:
        return jax.tree_map(lambda z: z[shift:], x)
    return jax.tree_map(lambda z: z[:shift], x)

def none_or_concat(x, y, position=1):
    if x is None or y is None:
        return None
    if position == 1:
        return jax.tree_map(lambda a, b: jnp.concatenate([a[None, ...], b]), y, x)
    else:
        return jax.tree_map(lambda a, b: jnp.concatenate([b, a[None, ...]]), y, x)
    
def mvn_loglikelihood(x, chol_cov):
    """multivariate normal"""
    dim = chol_cov.shape[0]
    y = jlinalg.solve_triangular(chol_cov, x, lower=True)
    normalizing_constant = (
            jnp.sum(jnp.log(jnp.abs(jnp.diag(chol_cov)))) + dim * jnp.log(2 * jnp.pi) / 2.0
    )
    norm_y = jnp.sum(y * y, -1)
    return -0.5 * norm_y - normalizing_constant