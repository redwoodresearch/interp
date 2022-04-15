from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
from jax.tree_util import PyTreeDef, tree_unflatten, tree_flatten, register_pytree_node_class, tree_map
from attrs import define, evolve, frozen

from interp.tools.indexer import I
from interp.tools.jax_tree_util import AttrsPartiallyStatic
from interp.tools.positive_definite import check_valid_positive_semi_def, make_sym
import interp.tools.optional as op


def flatten_leaves(leaves):
    return jnp.concatenate([x.flatten() for x in leaves])


# just a wrapper around a tuple to avoid pytree flattening
class ShapeTuple(tuple):
    ...


@frozen
class FlatValueConfig:
    tree_def: PyTreeDef = tree_flatten(object())[1]
    shapes: Optional[Tuple[Tuple[int, ...], ...]] = None

    def as_shaped_list_gen(self, x: jnp.ndarray, dims: Tuple[int, ...], reshape: bool = True) -> List[jnp.ndarray]:
        i_lst = list(range(x.ndim))
        dims = tuple(i_lst[d] for d in dims)
        assert len(set(x.shape[d] for d in dims)) == 1
        if self.shapes is None:
            return [x]
        else:
            out = []
            remaining = x
            for shape in self.shapes:
                assert all(map(lambda n: n >= 0, shape))
                size = math.prod(shape)
                val = remaining[tuple(I[:size] if d in dims else I[:] for d in range(x.ndim))]
                if reshape:
                    val = val.reshape(*sum((shape if d in dims else (x.shape[d],) for d in range(x.ndim)), ()))
                out.append(val)
                remaining = remaining[tuple(I[size:] if d in dims else I[:] for d in range(x.ndim))]

            assert remaining.size == 0

            return out

    def sizes_pre_at(self, idx: Tuple[int, ...]) -> List[Tuple[int, int]]:
        assert self.shapes is not None
        return [(sum(map(math.prod, self.shapes[:i])), math.prod(self.shapes[i])) for i in idx]

    def full_for_idx(self, idx: Tuple[int, ...]) -> Tuple[slice, ...]:
        return tuple(I[pre : pre + at] for pre, at in self.sizes_pre_at(idx))

    def set_at_idx(self, x: jnp.ndarray, y: jnp.ndarray, idx: Tuple[int, ...]) -> jnp.ndarray:
        return x.at[self.full_for_idx(idx)].set(y)

    def get_at_idx(self, x: jnp.ndarray, idx: Tuple[int, ...]) -> jnp.ndarray:
        return x[self.full_for_idx(idx)]

    def as_shaped_list(self, x: jnp.ndarray, dim: int = -1, reshape: bool = True) -> List[jnp.ndarray]:
        return self.as_shaped_list_gen(x, dims=(dim,), reshape=reshape)

    def as_diagonal_blocks(
        self, x: jnp.ndarray, dims: Tuple[int, int] = (-2, -1), reshape: bool = True
    ) -> List[jnp.ndarray]:
        return self.as_shaped_list_gen(x, dims=dims, reshape=reshape)

    def as_diag_blocks_tree(self, x: jnp.ndarray, dims: Tuple[int, int] = (-2, -1)) -> Any:
        return tree_unflatten(self.tree_def, self.as_diagonal_blocks(x, dims))

    def as_tree(self, x: jnp.ndarray, dim: int = -1, reshape: bool = True) -> Any:
        return self.as_nested_tree(x, dims=(dim,), reshape=reshape)

    def as_nested_tree(self, x: jnp.ndarray, dims: Tuple[int, ...] = (-1,), reshape: bool = True):
        tree: Any = x
        for dim in dims:
            tree = tree_map(
                lambda x: tree_unflatten(self.tree_def, self.as_shaped_list(x, dim=dim, reshape=reshape)), tree
            )

        return tree

    def shape_tree(self):
        return tree_map(lambda x: x[0], self.nested_shape_tree(1))

    def nested_shape_tree(self, n: int):
        tree = ShapeTuple(())
        for _ in range(n):
            tree = tree_map(
                lambda s_l: tree_unflatten(self.tree_def, (ShapeTuple(s_l + (s_r,)) for s_r in op.unwrap(self.shapes))),
                tree,
            )

        return tree

    def as_diagonal_blocks_tree(self, x: jnp.ndarray, dims: Tuple[int, int] = (-2, -1)) -> Any:
        return tree_unflatten(self.tree_def, self.as_diagonal_blocks(x, dims=dims))

    @classmethod
    def from_tree(cls, x: Any) -> Tuple[FlatValueConfig, jnp.ndarray]:
        x = tree_map(jnp.asarray, x)  # this allows for having scalars in your tree
        leaves, tree_def = tree_flatten(x)
        return FlatValueConfig(tree_def, tuple(l.shape for l in leaves)), flatten_leaves(leaves)

    def idxs_tree_def_for_selector(self, selector: Callable[[Any], Any]) -> Tuple[List[int], PyTreeDef]:
        idxs = [0] if self.shapes is None else list(range(len(self.shapes)))
        out_idxs, tree_def = tree_flatten(selector(tree_unflatten(self.tree_def, idxs)))

        out_final = []
        for idx in out_idxs:
            assert isinstance(idx, int)
            assert 0 <= idx < len(idxs)
            out_final.append(idx)

        return out_final, tree_def

    def other_idxs(self, idxs: Iterable[int]):
        assert self.shapes is not None
        return list(sorted(set(range(len(self.shapes))).difference(idxs)))

    def idx_select(self, x: jnp.ndarray, idxs: List[int], reshape: bool = True):
        lst = self.as_shaped_list(x, reshape=reshape)
        assert idxs is not None
        return [lst[i] for i in idxs]

    def replace_idx_from(self, to: jnp.ndarray, idx_vals: List[Tuple[int, jnp.ndarray]]):
        lst = self.as_shaped_list(to, reshape=False)
        for idx, val in idx_vals:
            lst[idx] = val

        return flatten_leaves(lst)


# handle type stuff
def diag(x) -> jnp.ndarray:
    out = jnp.diag(x)
    assert not isinstance(out, tuple)
    return out


def spherical_approximation(covariance: jnp.ndarray, _: Any = None):
    trace = diag(covariance).sum()

    return jnp.eye(covariance.shape[0]) * trace / covariance.shape[0]


def zero_approximation(covariance: jnp.ndarray, _: Any = None):
    return jnp.zeros_like(covariance)


@register_pytree_node_class
@define
class MultivariateNormal(AttrsPartiallyStatic):
    mean: jnp.ndarray
    covariance: jnp.ndarray
    flat_value_config: FlatValueConfig = FlatValueConfig()
    sampling_eps: Union[jnp.ndarray, float] = 1e-5
    always_check_valid: bool = False
    check_pos_def: bool = False
    approximation: Callable[[jnp.ndarray, FlatValueConfig], jnp.ndarray] = lambda x, _: x
    maybe_invalid_shape: bool = False

    TreeOp = Callable[[Any], Any]
    Op = Callable[[jnp.ndarray], jnp.ndarray]

    def __attrs_post_init__(self):
        if self.maybe_invalid_shape and (self.mean.ndim != 1 or isinstance(self.mean, jax.ShapeDtypeStruct)):
            return

        self.check_shape()

        if self.always_check_valid:
            self.check_valid(check_pos_def=self.check_pos_def)

    def non_static_names(self) -> Set[str]:
        return {"mean", "covariance", "sampling_eps"}

    def needs_deep_copy(self) -> bool:
        return False

    @property
    def size(self):
        return self.mean.size

    @property
    def dtype(self):
        return self.mean.dtype

    def mean_as(self):
        return self.flat_value_config.as_tree(self.mean)

    def covariance_as(self):
        return self.flat_value_config.as_nested_tree(self.covariance, dims=(-2, -1))

    def check_shape(self):
        assert self.mean.dtype == self.covariance.dtype

        assert self.mean.ndim == 1
        assert self.covariance.shape == (self.size, self.size)

        # just to potentially trip asserts
        self.mean_as()
        self.covariance_as()

    def check_valid(self, check_pos_def: bool = False):
        self.check_shape()
        assert not jnp.isnan(self.mean).any()
        assert not jnp.isnan(self.covariance).any()
        if check_pos_def:
            check_valid_positive_semi_def(self.patched_cov(self.covariance, eps=self.sampling_eps))

    def normal_like(
        self, mean: jnp.ndarray, covariance: jnp.ndarray, flat_value_config: Optional[FlatValueConfig] = None
    ):
        """
        Create a new normal with the same configuration and apply the approximation (if any).
        """

        flat_value_config = op.unwrap_or(flat_value_config, self.flat_value_config)
        return evolve(
            self,
            mean=mean,
            covariance=self.approximation(covariance, flat_value_config),
            flat_value_config=flat_value_config,
        )

    @staticmethod
    def patched_cov(cov: jnp.ndarray, eps=0.0):
        variance = diag(cov)
        size = cov.shape[0]
        return (
            cov.at[jnp.arange(size), jnp.arange(size)].set(
                jnp.where(variance == 0.0, jnp.ones_like(cov, shape=(), dtype=variance.dtype), variance)
            )
            + jnp.eye(size, dtype=variance.dtype) * eps
        )

    # very scary function
    def sample(self, key: jax.random.KeyArray, shape: Tuple[int, ...]) -> jnp.ndarray:
        patched_cov = self.patched_cov(self.covariance, eps=self.sampling_eps)

        samples: jnp.ndarray = jax.lax.cond(
            jnp.isnan(patched_cov).any(),
            lambda: jnp.full(shape + (self.size,), float("nan"), dtype=patched_cov.dtype),
            lambda: jax.random.multivariate_normal(key, self.mean, patched_cov, shape, dtype=patched_cov.dtype),
        )

        return jnp.where(diag(self.covariance) == 0.0, self.mean, samples)

    def op_get(self, op: TreeOp, x: jnp.ndarray):
        return FlatValueConfig.from_tree(op(self.flat_value_config.as_tree(x)))

    def wrap_get(self, op: TreeOp):
        return lambda x: self.op_get(op, x)

    def lin_op(self, linear_operator: TreeOp) -> MultivariateNormal:
        """
        linear_operator must be actually linear (not affine!) for correctness.
        """

        get = self.wrap_get(linear_operator)
        new_config, new_mean = get(self.mean)

        get_vals = lambda x: get(x)[1]

        new_covariance_0 = jax.vmap(get_vals, in_axes=1, out_axes=1)(self.covariance)
        new_covariance = jax.vmap(get_vals)(new_covariance_0)
        new_covariance = make_sym(new_covariance)

        return self.normal_like(new_mean, new_covariance, flat_value_config=new_config)

    def combine_dist(
        self,
        combine: Callable[[Any, Any], Any],
        new_new_cov,
        orig_new_cov,
        new_means,
        new_config: Optional[FlatValueConfig] = None,
    ):
        """
        Suppose we have function f and let's call the current multivariate normal `Orig`.

        We'll call `New = f(Orig)`.

        This function computes the multivariate normal distribution
        corresponding to `combine(f(Orig), Orig)`.
        """

        combine_overall = lambda new, orig: FlatValueConfig.from_tree(
            combine(new if new_config is None else new_config.as_tree(new), self.flat_value_config.as_tree(orig))
        )

        combined_config, overall_means = combine_overall(new_means, self.mean)
        combine_v = lambda new, orig: combine_overall(new, orig)[1]

        combine_new = jax.vmap(combine_v, in_axes=1, out_axes=1)(new_new_cov, orig_new_cov)
        orig_combine = jax.vmap(combine_v)(orig_new_cov, self.covariance)
        combine_combine = jax.vmap(combine_v, in_axes=(0, 1))(combine_new, orig_combine)

        return self.normal_like(overall_means, combine_combine, flat_value_config=combined_config)

    def get_beta(
        self, samples: jnp.ndarray, vals: jnp.ndarray, sample_selector: TreeOp, weights: Optional[jnp.ndarray] = None
    ):
        """
        Primarily a helper function for Monte Carlo sampling.
        Performs a linear regression from samples to values (subtracting off the expectation from each)
        """
        n_samples = samples.shape[0]
        assert n_samples == vals.shape[0]

        selected_dist = self.lin_op(sample_selector)

        vals_mean = jnp.average(vals, axis=0, weights=weights)

        def run_lstsq():
            x = samples - selected_dist.mean
            y = vals - vals_mean[None]
            if weights is not None:
                w = jnp.sqrt(weights[:, None])
                x *= w
                y *= w

            return jnp.linalg.lstsq(x, y, rcond=None)[0]

        beta = jax.lax.cond(
            jnp.isnan(vals).any() | jnp.isnan(samples).any(),
            lambda: jnp.full((selected_dist.mean.size, vals_mean.size), float("nan"), dtype=samples.dtype),
            run_lstsq,
        )
        beta = jnp.where(jnp.expand_dims(diag(selected_dist.covariance) == 0, 1), jnp.zeros_like(beta, shape=()), beta)

        return beta

    def get_beta_orig_new(
        self, samples: jnp.ndarray, new: jnp.ndarray, sample_selector: TreeOp, weights: Optional[jnp.ndarray] = None
    ):
        """
        Primarily a helper for Monte Carlo sampling .
        Uses a linear regression estimator (beta) and the original covariance to estimate the covariance
        between original and new.
        """
        beta = self.get_beta(samples, new, sample_selector, weights=weights)

        select = lambda x: self.wrap_get(sample_selector)(x)[1]
        orig_new = jax.vmap(lambda x: jnp.einsum("n m, n -> m", beta, select(x)))(self.covariance)

        return beta, orig_new

    def monte_carlo_non_linearity_samples(
        self,
        key: jax.random.KeyArray,
        f: Callable[[Any, jax.random.KeyArray], Any],
        iters: int,
        samples: jnp.ndarray,
        sample_selector: TreeOp = lambda x: x,
        combine: Callable[[Any, Any], Any] = lambda new, _: new,
        weights: Optional[jnp.ndarray] = None,
    ) -> MultivariateNormal:
        """
        See docs for monte_carlo_non_linearity.

        This function takes samples (and optionally weights) as arguments
        instead of doing the sampling itself. Can be used for importance sampling.
        """
        selected_dist = self.lin_op(sample_selector)

        new_config = None

        def call_on_full(sample_key):
            sample, key = sample_key
            nonlocal new_config
            new_config, out = selected_dist.wrap_get(lambda x: f(x, key))(sample)
            return out

        subkeys = jax.random.split(key, iters)
        ret_vals = jax.vmap(call_on_full)((samples, subkeys))
        assert new_config is not None

        new_mean = jnp.average(ret_vals, axis=0, weights=weights)

        # beta is results of linear regression from orig -> new
        beta, orig_new_cov = self.get_beta_orig_new(samples, ret_vals, sample_selector, weights=weights)

        new_new_shape = (ret_vals.shape[1], ret_vals.shape[1])

        # calculate covariance between new and new
        if iters == 1:
            new_new_cov = jnp.zeros_like(ret_vals, shape=new_new_shape)
        else:
            # decompose covariance into covariance implied by the linear regression predictions and
            # extra covariance due to difference between true samples and prediction
            # (this method ensures that the new new covariance is consistent with orig new covariance, since that
            # is also calculated via the regression)
            predicted_new = (
                jnp.einsum("i n, n m -> i m", samples - jnp.expand_dims(selected_dist.mean, 0), beta) + new_mean[None]
            )
            new_new_cov_extra = jnp.broadcast_to(
                jnp.cov(ret_vals - predicted_new, rowvar=False, aweights=weights), new_new_shape
            )
            beta_implied_new_new = jnp.einsum("n N, n m, N M -> m M", selected_dist.covariance, beta, beta)
            new_new_cov = make_sym(new_new_cov_extra + beta_implied_new_new)

        return self.combine_dist(combine, new_new_cov, orig_new_cov, new_means=new_mean, new_config=new_config)

    # it's probably possible to do much better than this by knowing some math facts about your nonlinearity
    def monte_carlo_non_linearity(
        self,
        key: jax.random.KeyArray,
        f: Callable[[Any, jax.random.KeyArray], Any],
        iters: int,
        sample_selector: TreeOp = lambda x: x,
        combine: Callable[[Any, Any], Any] = lambda new, _: new,
    ) -> MultivariateNormal:
        """
        Compute mean and covariance of the distribution after a (possibly
        non-linear) function via monte-carlo integration.

        sample_selector must be linear and it selects values to actually be sampled

        combine constructs a new distribution given the newly produced values
        and the original values. It's also required to be linear.

        (TODO: I haven't actually tested the case where combine doesn't just
        permute, but this should be fine)
        """

        selected_dist = self.lin_op(sample_selector)

        assert iters >= 1
        key, subkey = jax.random.split(key)
        samples = selected_dist.sample(subkey, (iters,))

        return self.monte_carlo_non_linearity_samples(
            key, f, iters=iters, samples=samples, sample_selector=sample_selector, combine=combine
        )

    # maybe should be merged with above, but does feel like weird special case...
    def monte_carlo_non_linearity_operate_gaussian(
        self,
        key: jax.random.KeyArray,
        f: Callable[[Any, jax.random.KeyArray], Any],
        operate_gaussian: Callable[[MultivariateNormal], MultivariateNormal],
        iters: int,
        sample_selector: TreeOp = lambda x: x,
        combine: Callable[[Any, Any], Any] = lambda new, _: new,
    ) -> MultivariateNormal:
        """
        See docs for monte_carlo_non_linearity.

        This function allows for importance sampling via modifying the sampled gaussian.
        """

        selected_dist = self.lin_op(sample_selector)
        op_selected_dist = operate_gaussian(selected_dist)

        assert selected_dist.flat_value_config == op_selected_dist.flat_value_config

        assert iters >= 1
        key, subkey = jax.random.split(key)
        samples = op_selected_dist.sample(subkey, (iters,))

        weights = jstats.multivariate_normal.pdf(
            samples, selected_dist.mean, self.patched_cov(selected_dist.covariance, self.sampling_eps)
        ) / jstats.multivariate_normal.pdf(
            samples, op_selected_dist.mean, self.patched_cov(op_selected_dist.covariance, self.sampling_eps)
        )

        return self.monte_carlo_non_linearity_samples(
            key,
            f,
            iters=iters,
            samples=samples,
            sample_selector=sample_selector,
            combine=combine,
            weights=weights,
        )

    # compare to numpy einsum
    def mul(
        self,
        flat_value_l: int,
        l_axes_names: List[Union[int, str]],
        flat_value_r: int,
        r_axes_names: List[Union[int, str]],
        out_axes_names: List[Union[int, str]],
        combine: Callable[[jnp.ndarray, Any], Any] = lambda new, _: new,
    ) -> MultivariateNormal:
        """
        Compute mean and covariance of the distribution after taking the product of two values (einsum style).

        combine constructs a new distribution given the newly produced values
        and the original values. It's required to be linear.

        (TODO: I haven't actually tested the case where combine doesn't just
        permute, but this should be fine)
        """

        assert self.flat_value_config.shapes is not None

        l_shape = self.flat_value_config.shapes[flat_value_l]
        r_shape = self.flat_value_config.shapes[flat_value_r]

        assert len(l_shape) == len(l_axes_names)
        assert len(r_shape) == len(r_axes_names)

        assert len(set(l_axes_names)) == len(l_axes_names)
        assert len(set(r_axes_names)) == len(r_axes_names)
        assert len(set(out_axes_names)) == len(out_axes_names)

        next_id = 0
        name_to_id: Dict[Union[int, str], int] = {}

        def to_ids(axes_names) -> List[int]:
            out = []
            for name in axes_names:
                if name in name_to_id:
                    out.append(name_to_id[name])
                else:
                    nonlocal next_id
                    this_id = next_id
                    out.append(this_id)
                    name_to_id[name] = this_id

                    next_id += 1

            return out

        l_ns = to_ids(l_axes_names)
        r_ns = to_ids(r_axes_names)
        o_ns = to_ids(out_axes_names)

        assert next_id < 500

        def get_snd(names):
            return [x + 1000 for x in names]

        s_l_ns = get_snd(l_ns)
        s_r_ns = get_snd(r_ns)
        s_o_ns = get_snd(o_ns)

        def get_select(is_l):
            return lambda x: self.flat_value_config.idx_select(
                x, [flat_value_l if is_l else flat_value_r], reshape=False
            )[0]

        l_means = get_select(True)(self.mean).reshape(l_shape)
        r_means = get_select(False)(self.mean).reshape(r_shape)

        def reshape_to(x, is_l_first, is_l_second):
            return jax.vmap(get_select(is_l_first), in_axes=1, out_axes=1,)(
                jax.vmap(get_select(is_l_second))(x)
            ).reshape(*(l_shape if is_l_first else r_shape), *(l_shape if is_l_second else r_shape))

        l_l_cov = reshape_to(self.covariance, True, True)
        r_r_cov = reshape_to(self.covariance, False, False)
        l_r_cov = reshape_to(self.covariance, True, False)

        orig_l_cov = jax.vmap(get_select(True))(self.covariance).reshape(self.mean.size, *l_shape)
        orig_r_cov = jax.vmap(get_select(False))(self.covariance).reshape(self.mean.size, *r_shape)

        l_r_means = jnp.einsum(l_means, l_ns, r_means, s_r_ns, l_ns + s_r_ns)
        l_l_means = jnp.einsum(l_means, l_ns, l_means, s_l_ns, l_ns + s_l_ns)

        new_means = jnp.einsum(l_r_means + l_r_cov, l_ns + r_ns, o_ns)
        new_shape = new_means.shape
        new_size = new_means.size

        out = o_ns + s_o_ns

        # magic formulas from the great beyond

        # I wonder if setting lower == upper will result in jax optimizing the einsum
        new_new_cov = (
            jnp.einsum(l_l_cov + l_l_means, l_ns + s_l_ns, r_r_cov, r_ns + s_r_ns, out)
            + jnp.einsum(r_means, r_ns, r_means, s_r_ns, l_l_cov, l_ns + s_l_ns, out)
            + jnp.einsum(l_r_cov, l_ns + s_r_ns, l_r_cov + l_r_means, s_l_ns + r_ns, out)
            + jnp.einsum(l_r_means, l_ns + s_r_ns, l_r_cov, s_l_ns + r_ns, out)
        )

        orig_ns = [3000]
        orig_out = orig_ns + o_ns

        orig_new_cov = jnp.einsum(l_means, l_ns, orig_r_cov, orig_ns + r_ns, orig_out) + jnp.einsum(
            r_means, r_ns, orig_l_cov, orig_ns + l_ns, orig_out
        )

        return self.combine_dist(
            lambda new, orig: combine(new.reshape(new_shape), orig),
            make_sym(new_new_cov.reshape(new_size, new_size)),
            orig_new_cov.reshape(self.size, new_size),
            new_means.flatten(),
        )

    def mul_select(
        self,
        selector_l: TreeOp,
        l_axes_names: List[Union[int, str]],
        selector_r: TreeOp,
        r_axes_names: List[Union[int, str]],
        out_axes_names: List[Union[int, str]],
        combine: Callable[[jnp.ndarray, Any], Any] = lambda new, _: new,
    ) -> MultivariateNormal:
        """
        Wrapper for mul which uses functions to select which items you're interested in.

        selectors must return exact 1 tensor with no operations applied:
        they're just for specifying which values to operate on.
        """

        def select(selector):
            idxs = self.flat_value_config.idxs_tree_def_for_selector(selector)[0]
            assert len(idxs) == 1
            return idxs[0]

        return self.mul(select(selector_l), l_axes_names, select(selector_r), r_axes_names, out_axes_names, combine)

    def add(self, add_func: TreeOp) -> MultivariateNormal:
        """
        add_func should promise very nicely to just add and NOTHING ELSE
        """

        config, new_mean = self.op_get(add_func, self.mean)
        assert config == self.flat_value_config
        assert new_mean.shape == self.mean.shape
        return self.normal_like(new_mean, self.covariance)

    def set(self, set_to: Any, setter: Callable[[Any, Any], Any]) -> MultivariateNormal:
        """
        set_to should be a pytree which set func takes as it's second argument

        set_items function should actually just set to the value of the second
        argument or great shame will be brought down upon your family

        We need to take a setting function because we also want to zero out some covariances.
        """

        get_setter = lambda to: self.wrap_get(lambda x: setter(x, to))

        config, new_mean = get_setter(set_to)(self.mean)

        assert config == self.flat_value_config
        assert new_mean.shape == self.mean.shape

        set_zero = lambda x: get_setter(tree_map(jnp.zeros_like, set_to))(x)[1]
        new_covariance = jax.vmap(set_zero, in_axes=1, out_axes=1)(jax.vmap(set_zero)(self.covariance))

        return self.normal_like(new_mean, new_covariance)

    @staticmethod
    def cov_exp_term(s_means, s_vars):
        return jnp.exp(s_means[None, :] + s_means[:, None] + (s_vars[None, :] + s_vars[:, None]) / 2)

    def exp(
        self,
        selector: TreeOp = lambda x: x,
        combine: Callable[[jnp.ndarray, Any], Any] = lambda new, _: new,
    ) -> MultivariateNormal:
        selected_dist = self.lin_op(selector)

        selected_vars = jnp.diag(selected_dist.covariance)

        # wikipedia formulas
        new_means = jnp.exp(selected_dist.mean + selected_vars / 2)
        new_new_cov = self.cov_exp_term(selected_dist.mean, selected_vars) * (jnp.exp(selected_dist.covariance) - 1.0)

        orig_selected_cov = jax.vmap(lambda x: self.wrap_get(selector)(x)[1])(self.covariance)
        orig_new_cov = jnp.einsum("o s, s -> o s", orig_selected_cov, new_means)  # magic formula

        return self.combine_dist(
            combine, new_new_cov, orig_new_cov, new_means, new_config=selected_dist.flat_value_config
        )

    # pretends was log-normal and removes exp
    # only does what you want if distribution is valid log-normal
    def log(
        self,
        selector: TreeOp = lambda x: x,
        combine: Callable[[jnp.ndarray, Any], Any] = lambda new, _: new,
    ) -> MultivariateNormal:
        # just inverse of above
        selected_dist = self.lin_op(selector)
        selected_vars = jnp.diag(selected_dist.covariance)

        # wikipedia formulas
        new_means = jnp.log(selected_dist.mean ** 2 / jnp.sqrt(selected_dist.mean ** 2 + selected_vars))
        new_vars = jnp.log(1 + selected_vars / selected_dist.mean ** 2)

        new_new_cov = jnp.log(selected_dist.covariance / self.cov_exp_term(new_means, new_vars) + 1.0)

        orig_selected_cov = jax.vmap(lambda x: self.wrap_get(selector)(x)[1])(self.covariance)
        orig_new_cov = orig_selected_cov / selected_dist.mean[None]

        return self.combine_dist(
            combine, new_new_cov, orig_new_cov, new_means, new_config=selected_dist.flat_value_config
        )

    def condition(
        self,
        selector: TreeOp,
        value: Any,
        setter: Optional[Callable[[Any, Any], Any]] = None,
    ) -> MultivariateNormal:
        """
        selector is allowed do any any linear operation.

        value must have same type + (broadcasts to) shape as output of this
        linear operation

        If selector just gets a subset of the distribution, you can pass a
        setter which must sets the same subset as is selected. This will
        improve numerical accuracy (by ensuring means are exact and covariances
        are zero for conditioned on values).

        The setter obeys the same spec as .set with the `set_to` type and shape
        coming from the result of `selector`.
        """

        return self.condition_mixture(
            selector,
            tree_map(lambda x: jnp.asarray(x)[None], value),
            jnp.ones((1,), dtype=self.mean.dtype),
            setter=setter,
        )

    # sources:
    # - https://math.stackexchange.com/questions/195911/calculation-of-the-covariance-of-gaussian-mixtures
    # - https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    def condition_mixture(
        self,
        selector: TreeOp,
        values: Any,
        weights: jnp.ndarray,
        setter: Optional[Callable[[Any, Any], Any]] = None,
    ) -> MultivariateNormal:
        """
        This computes the means and covariances for a mixture of a bunch of
        conditional distributions with different weights.

        selector is allowed do any any linear operation.

        value must have same type + (broadcasts to) shape as output of this
        linear operation

        weights must be >= 0, but don't have to be normalized

        If selector just gets a subset of the distribution, you can pass a
        setter which must sets the same subset as is selected. This will
        improve numerical accuracy (by ensuring means are exact and covariances
        are zero for conditioned on values).

        The setter obeys the same spec as .set with the `set_to` type and shape
        coming from the result of `selector`.
        """

        assert weights.ndim == 1

        def check_weight_dim_eq(x):
            assert x.shape[0] == weights.shape[0]

        tree_map(check_weight_dim_eq, values)

        weights = weights / weights.sum()

        selected_dist = self.lin_op(selector)

        def from_tree(x):
            select_config, out = FlatValueConfig.from_tree(
                jax.tree_map(lambda x, s: jnp.broadcast_to(x, s.shape), x, selected_dist.mean_as())
            )
            assert select_config == selected_dist.flat_value_config
            return out

        flat_values = jax.vmap(from_tree)(values)

        orig_selected = jax.vmap(lambda x: self.wrap_get(selector)(x)[1])(self.covariance)

        inv_selected = jnp.linalg.pinv(selected_dist.covariance)

        new_means = self.mean[None] + jnp.einsum(
            "k n, n N, c N -> c k", orig_selected, inv_selected, flat_values - selected_dist.mean[None]
        )
        new_covariance = make_sym(
            self.covariance - jnp.einsum("k n, n N, K N -> k K", orig_selected, inv_selected, orig_selected)
        )

        new_mean = jnp.einsum("c, c k -> k", weights, new_means)

        base_out = self.normal_like(
            new_mean,
            new_covariance,
            evolve(self.flat_value_config, shapes=(self.mean.shape,))
            if self.flat_value_config.shapes is None
            else self.flat_value_config,
        )

        if setter is not None:
            # avoids accumulating numerical error
            flat_value = jnp.einsum("c, c n -> n", weights, flat_values)
            base_out = base_out.set(selected_dist.flat_value_config.as_tree(flat_value), setter)

        if weights.shape[0] == 1:
            return base_out
        else:
            if setter is not None:
                # set not technically needed, again for accuracy or whatever
                new_means = jax.vmap(
                    lambda x, to: self.op_get(
                        lambda x: op.unwrap(setter)(x, selected_dist.flat_value_config.as_tree(to)), x
                    )[1]
                )(new_means, flat_values)

            new_means_central = new_means - base_out.mean[None]
            return base_out.normal_like(
                base_out.mean,
                make_sym(
                    base_out.covariance
                    + jnp.einsum("c, c k, c K -> k K", weights, new_means_central, new_means_central)
                ),
            )


def empty_normal():
    return MultivariateNormal(jnp.zeros((0,)), jnp.zeros((0, 0)))
