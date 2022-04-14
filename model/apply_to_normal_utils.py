from typing import Tuple, Callable

import jax
import jax.numpy as jnp

from interp.tools.indexer import I
from interp.tools.multivariate_normal import MultivariateNormal, FlatValueConfig
from interp.tools.immutable_dict import operate_f, operate


def run_per_dist(dist: MultivariateNormal, get_func):
    mean_as = dist.mean_as()

    if isinstance(mean_as, dict):
        assert set(mean_as.keys()) == {"final_embeds", "input_embeds"}
        func = get_func(mean_as["final_embeds"].shape[1])
        return dist.lin_op(
            operate_f("final_embeds", "final_embeds", func) @ operate_f("input_embeds", "input_embeds", func)
        )
    else:
        func = get_func(mean_as.shape[1])
        return dist.lin_op(func)


def pad_dist(dist: MultivariateNormal, pad_to: int):
    def get_pad_item(seq_len):
        assert seq_len <= pad_to
        return lambda x: jnp.concatenate(
            [x, jnp.zeros_like(x, shape=(x.shape[0], pad_to - seq_len, x.shape[2]))], axis=1
        )

    return run_per_dist(dist, get_pad_item)


def un_pad_dist(dist: MultivariateNormal, seq_len: int):
    return run_per_dist(dist, lambda _: lambda x: x[:, :seq_len])


def map_dict_keys_covariance(
    covariance: jnp.ndarray,
    config: FlatValueConfig,
    f: Callable[[str, str, Tuple[int, ...], Tuple[int, ...], jnp.ndarray], jnp.ndarray],
):
    tree = config.as_nested_tree(covariance, dims=(-2, -1), reshape=False)
    shapes = config.shape_tree()
    if isinstance(tree, dict):
        out_cov = jax.vmap(lambda x: config.from_tree(x)[1], in_axes=1, out_axes=1)(
            {
                k_l: jax.vmap(lambda x: config.from_tree(x)[1], in_axes=0, out_axes=0)(
                    {k_r: f(k_l, k_r, shapes[k_l], shapes[k_r], v) for k_r, v in sub_tree.items()}
                )
                for k_l, sub_tree in tree.items()
            }
        )

        return out_cov

    else:
        return covariance


def graft_input_embed(dist, loc, desired_mean, desired_cov):
    def map_func(k_l, k_r, s_l, s_r, current):
        if k_l != "input_embeds" and k_r != "input_embeds":
            return current

        new = current.reshape(*s_l, *s_r)

        if k_l == "input_embeds":
            new = new.at[:, loc].set(0.0)
        if k_r == "input_embeds":
            new = new.at[(I[:],) * (len(s_l) + 1) + (loc,)].set(0.0)
        if k_l == "input_embeds" and k_r == "input_embeds":
            new = new.at[0, loc, :, 0, loc, :].set(desired_cov)

        return new.reshape(*current.shape)

    replaced_cov = map_dict_keys_covariance(dist.covariance, dist.flat_value_config, map_func)
    new_config, replaced_mean = FlatValueConfig.from_tree(
        operate(
            dist.mean_as(),
            "input_embeds",
            "input_embeds",
            lambda x: x.at[0, loc].set(desired_mean),
        )
    )
    assert new_config == dist.flat_value_config

    return dist.normal_like(replaced_mean, replaced_cov)
