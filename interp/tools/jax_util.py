from typing import Any, List, Optional, Union

import jax
from jax import lax
import jax.numpy as jnp

import interp.tools.optional as op


def sum_iter_nonempty_allow_none(it):
    out = None
    for item in it:
        if out is None:
            out = item
        else:
            out = out + item

    return out


def sum_iter_nonempty(it):
    out = sum_iter_nonempty_allow_none(it)
    assert out is not None
    return out


def stack_tree(vals: List[Any], stack=True, axis=0) -> Any:
    """
    cat/stack over list dim using tree_map
    """

    return jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=axis) if stack else jnp.concatenate(xs, axis=axis), *vals
    )


def maybe_static_cond(cond: Union[bool, jnp.ndarray], l, r, *args, force_is_static: Optional[bool] = None):
    is_static = op.unwrap_or(force_is_static, isinstance(cond, bool))
    if is_static:
        if cond:
            return l(*args)
        else:
            return r(*args)
    else:
        return lax.cond(cond, l, r, *args)
