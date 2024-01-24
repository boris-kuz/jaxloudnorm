# note: to be removed from here once I move it to its own package

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.signal import convolve
from jax._src.util import canonicalize_axis
from jax._src.api_util import _ensure_index_tuple

from jaxtyping import ArrayLike, Array

from typing import Sequence, Set


def _axes_to_mapped_axes(axes: Sequence[int] | int, ndim: int) -> Set[int]:
    axes = _ensure_index_tuple(axes)
    axes = tuple(canonicalize_axis(ax, ndim) for ax in axes)
    return set(range(ndim)) - set(axes)


def lfilter(
    b: ArrayLike, a: ArrayLike, x: ArrayLike, axis: int = -1, zi: Array | None = None
) -> Array | tuple[Array, Array]:
    a_: Array = jnp.atleast_1d(jnp.asarray(a))
    b_: Array = jnp.atleast_1d(jnp.asarray(b))
    x_: Array = jnp.atleast_1d(jnp.asarray(x))

    if len(a_) == 1:
        b_ /= a_
        out_full = jnp.apply_along_axis(
            lambda y: convolve(b_, y, mode="full"), axis, x_
        )

        ind = out_full.ndim * [slice(None)]

        if zi is not None:
            ind[axis] = slice(zi.shape[axis])
            out_full = out_full.at[tuple(ind)].set(out_full[tuple(ind)] + zi)

        ind[axis] = slice(out_full.shape[axis] - len(b_) + 1)
        out = out_full[tuple(ind)]

        if zi is None:
            return out
        else:
            ind[axis] = slice(out_full.shape[axis] - len(b_) + 1, None)
            zf = out_full[tuple(ind)]
            return out, zf

    else:
        max_coef_len = max(len(b_), len(a_))

        b_, a_ = (
            x if len(x) == max_coef_len else jnp.pad(x, (0, max_coef_len - len(x)))
            for x in (b_, a_)
        )

        b_, a_ = (x / a_[0] for x in (b_, a_))

        _lfilter = lambda x, zi: _lfilter_unbatched(
            b_,
            a_,
            x,
            zi=zi if zi is not None else jnp.zeros_like(x, shape=max_coef_len - 1),
        )

        mapped_axes = _axes_to_mapped_axes(axis, x_.ndim)

        for axis in sorted(mapped_axes):
            _lfilter = jax.vmap(_lfilter, in_axes=axis, out_axes=axis)
        out, zi_ = _lfilter(x_, zi)

        if zi is None:
            return out
        else:
            return out, zi_


def _lfilter_unbatched(b: Array, a: Array, x: Array, zi: Array) -> tuple[Array, Array]:
    def f(state: Array, x_: Array) -> tuple[Array, Array]:
        y = state[0] + b[0] * x_

        def calc_middle_delays(_, a_b_z_: Array) -> tuple[Array, Array]:
            a_, b_, z_ = a_b_z_
            z = z_ + x_ * b_ - y * a_
            return z, z

        middle_states = lax.scan(
            calc_middle_delays,
            state[1],
            jnp.stack([a[1:-1], b[1:-1], state[1:]], axis=1),
        )[1]

        last_state = x_ * b[-1] - y * a[-1]
        return jnp.concatenate([middle_states, last_state.reshape(-1)]), y

    zi, out = lax.scan(f, zi, x)

    return out, zi
