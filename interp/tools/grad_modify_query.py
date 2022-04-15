from __future__ import annotations

from typing import Any, Callable, List, Protocol, Tuple, TypeVar, Union, Optional, Dict
from collections import defaultdict
from copy import copy

from attrs import define, Factory, frozen, evolve
import jax
from jax.interpreters.ad import JVPTracer
import jax.numpy as jnp
from interp.tools.jax_tree_util import default_is_leaf

from interp.tools.log import Logger, LoggerCacheAndModify, ModInfo, KeyIdxs, LogCache
import interp.tools.optional as op
from interp.tools.jax_util import stack_tree, sum_iter_nonempty, sum_iter_nonempty_allow_none


T = TypeVar("T")
ListStack = Union[T, List[T], List[List[T]]]


class GradModifierConf(Protocol):
    def get_mod(self, my_val: JVPTracer, all_vals: Dict[GradModifierConf, JVPTracer]) -> Optional[ModInfo]:
        ...

    def get_val(self) -> Optional[Union[jnp.ndarray, float]]:
        ...

    # TODO: maybe this should get moved outside protocol?
    def is_positive(self) -> bool:
        return True

    # python am.i.right
    def forget_type(self) -> GradModifierConf:
        return self


@define
class ModifierCollectionTreeNode:
    # none does nothing, can be nice if you aren't doing reduction
    # inner dim is always reduced over
    # doesn't make sense to only reduce over None
    confs: ListStack[GradModifierConf]
    next_item: ModifierCollectionTreeNodeStack = None

    # setting this changes the default for all descendants (who can override
    # this if desired)
    use_fwd: Optional[bool] = None

    positive: bool = True

    def term(self) -> ModifierCollectionTreeNode:
        return evolve(self, next_item=None)

    def next_item_node_check(self) -> ModifierCollectionTreeNode:
        return check_modifier_collection_tree_node(self.next_item)

    def next_item_list_node_check(self) -> List[ModifierCollectionTreeNode]:
        return check_list_modifier_collection_tree_node(self.next_item)

    def next_item_list_list_node_check(self) -> List[List[ModifierCollectionTreeNode]]:
        return check_list_list_modifier_collection_tree_node(self.next_item)

    def next_item_node_op_check(self) -> Optional[ModifierCollectionTreeNode]:
        return check_modifier_collection_tree_node_op(self.next_item)

    def next_item_list_node_op_check(self) -> List[Optional[ModifierCollectionTreeNode]]:
        return check_list_modifier_collection_tree_node_op(self.next_item)

    def next_item_list_list_node_op_check(self) -> List[List[Optional[ModifierCollectionTreeNode]]]:
        return check_list_list_modifier_collection_tree_node_op(self.next_item)


# maybe we shouldn't allow nones throughout stack? (makes some type stuff harder)
ModifierCollectionTreeNodeStack = ListStack[Optional[ModifierCollectionTreeNode]]


def check_modifier_collection_tree_node(x: ModifierCollectionTreeNodeStack) -> ModifierCollectionTreeNode:
    assert isinstance(x, ModifierCollectionTreeNode)
    return x


def check_list_modifier_collection_tree_node(xs: ModifierCollectionTreeNodeStack) -> List[ModifierCollectionTreeNode]:
    assert isinstance(xs, list)
    return [check_modifier_collection_tree_node(x) for x in xs]


def check_list_list_modifier_collection_tree_node(
    xs: ModifierCollectionTreeNodeStack,
) -> List[List[ModifierCollectionTreeNode]]:
    assert isinstance(xs, list)
    return [check_list_modifier_collection_tree_node(x) for x in xs]


def check_modifier_collection_tree_node_op(x: ModifierCollectionTreeNodeStack) -> Optional[ModifierCollectionTreeNode]:
    assert isinstance(x, ModifierCollectionTreeNode) or x is None
    return x


def check_list_modifier_collection_tree_node_op(
    xs: ModifierCollectionTreeNodeStack,
) -> List[Optional[ModifierCollectionTreeNode]]:
    assert isinstance(xs, list)
    return [check_modifier_collection_tree_node_op(x) for x in xs]


def check_list_list_modifier_collection_tree_node_op(
    xs: ModifierCollectionTreeNodeStack,
) -> List[List[Optional[ModifierCollectionTreeNode]]]:
    assert isinstance(xs, list)
    return [check_list_modifier_collection_tree_node_op(x) for x in xs]


@frozen
class TargetConf:
    log_key_idxs: KeyIdxs
    display_name: str = Factory(lambda self: self.log_key_idxs.key, takes_self=True)
    idx: Any = ...
    select_idx: Callable[[Any, Any], Any] = lambda v, idx: v[idx]

    def select(self, v):
        return self.select_idx(v, self.idx)


@define
class Query:
    targets: List[TargetConf]
    modifier_collection_tree: ModifierCollectionTreeNodeStack = None

    use_fwd: Optional[bool] = None
    allow_non_first_deriv: bool = False


def flatten_list_stack(l: ListStack[T]) -> Tuple[List[T], bool, int, defaultdict[int, int]]:
    out: List[T] = []
    add_dim = False
    num_buckets = 1
    flat_idx_to_bucket = defaultdict(int)
    if isinstance(l, list):
        assert len(l) > 0
        if isinstance(l[0], list):
            add_dim = True
            num_buckets = len(l)
            for bucket, xs in enumerate(l):
                assert isinstance(xs, list), "not all types the same!"
                for x in xs:
                    flat_idx_to_bucket[len(out)] = bucket
                    out.append(x)  # type: ignore
        else:
            for x in l:
                out.append(x)  # type: ignore
    else:
        out.append(l)

    assert len(out) > 0

    return out, add_dim, num_buckets, flat_idx_to_bucket


def stack_buckets(buckets: defaultdict[int, Any], num_buckets: int, add_dim: bool) -> Any:
    stacked = stack_tree(list(buckets[b] for b in range(num_buckets)), axis=-1)
    if not add_dim:
        stacked = jax.tree_util.tree_map(lambda x: x.squeeze(-1), stacked)
    return stacked


class Loggable(Protocol):
    def __call__(self, logger: Logger) -> Any:
        ...


def run_query(
    loggable: Loggable,
    query: Query,
    # copied each time, so effectively frozen...
    conf_vals: Dict[GradModifierConf, JVPTracer] = {},
    rec_call_from_deriv: bool = False,
) -> Dict[str, jnp.ndarray]:
    running_ret_vals: defaultdict[int, Optional[Dict[str, jnp.ndarray]]] = defaultdict(lambda: None)
    flattened_node, add_dim_node, num_buckets_node, flat_idx_to_bucket_node = flatten_list_stack(
        query.modifier_collection_tree
    )
    for i, modifier_collection_tree_node in enumerate(flattened_node):
        if modifier_collection_tree_node is None:
            logger = LoggerCacheAndModify()
            for t in query.targets:
                logger.add(t.log_key_idxs)
            for conf, val in conf_vals.items():
                mod_info = conf.get_mod(val, conf_vals)
                if mod_info is not None:
                    logger.add_mod(mod_info)
            cache: LogCache = loggable(logger=logger)
            out = {t.display_name: t.select(cache.get(t.log_key_idxs)) for t in query.targets}
        else:
            flattened_confs, add_dim_conf, num_buckets_conf, flat_idx_to_bucket_conf = flatten_list_stack(
                modifier_collection_tree_node.confs
            )
            conf_vals_this_iter = copy(conf_vals)

            is_deriv = any(conf.get_val() is not None for conf in flattened_confs)
            if rec_call_from_deriv and is_deriv:
                assert query.allow_non_first_deriv, "is non first deriv!"

            def func(vals):
                running_mul = 1.0
                assert len(vals) == len(flattened_confs)
                for conf, val in zip(flattened_confs, vals):
                    if conf.get_val() is None:
                        running_mul = running_mul + val
                    else:
                        assert conf not in conf_vals_this_iter, "duplicate conf along single path"

                    conf_vals_this_iter[conf] = val

                assert modifier_collection_tree_node is not None
                return jax.tree_map(
                    lambda v: running_mul * v,
                    run_query(
                        loggable,
                        evolve(query, modifier_collection_tree=modifier_collection_tree_node.next_item),
                        conf_vals=conf_vals_this_iter,
                        rec_call_from_deriv=is_deriv or rec_call_from_deriv,
                    ),
                )

            if is_deriv:
                use_fwd = op.unwrap_or(op.or_y(modifier_collection_tree_node.use_fwd, query.use_fwd), True)
                wrapped_func = (jax.jacfwd if use_fwd else jax.jacrev)(func)  # type: ignore[operator]
            else:
                wrapped_func = lambda vals: jax.tree_map(lambda x: [x for _ in vals], func(vals))

            unreduced_out: Dict[str, List[jnp.ndarray]] = wrapped_func(
                [op.unwrap_or(conf.get_val(), 0.0) for conf in flattened_confs]
            )

            def sum_and_bucket(xs):
                buckets: defaultdict[int, Optional[jnp.ndarray]] = defaultdict(lambda: None)
                for j, (x, conf) in enumerate(zip(xs, flattened_confs)):
                    conf_bucket = flat_idx_to_bucket_conf[j]
                    buckets[conf_bucket] = sum_iter_nonempty_allow_none(
                        [buckets[conf_bucket], x if conf.is_positive() else -x]
                    )

                return stack_buckets(buckets, num_buckets_conf, add_dim_conf)

            out = {k: sum_and_bucket(v) for k, v in unreduced_out.items()}

        modifier_collection_tree_bucket = flat_idx_to_bucket_node[i]
        running_ret_vals[modifier_collection_tree_bucket] = jax.tree_util.tree_map(
            lambda x, *running: sum_iter_nonempty(
                [x if modifier_collection_tree_node is None or modifier_collection_tree_node.positive else -x, *running]
            ),
            out,
            *op.it(running_ret_vals[modifier_collection_tree_bucket])
        )

    # hmm, might need to use aux data!
    return stack_buckets(running_ret_vals, num_buckets_node, add_dim_node)


def run_queries(
    loggable: Loggable,
    queries: Dict[str, Query],
    use_fwd: Optional[bool] = None,
) -> Dict[str, Dict[str, jnp.ndarray]]:
    return {
        n: run_query(
            loggable,
            evolve(query, use_fwd=op.or_y(query.use_fwd, use_fwd)),
        )
        for n, query in queries.items()
    }
