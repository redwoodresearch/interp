from typing import Any, Dict, Set
from copy import deepcopy

from jax.tree_util import tree_flatten, tree_unflatten, treedef_is_leaf
from attrs import define, asdict


def flatten_partially_static(items: Dict[str, Any], non_static_names: Set[str], needs_deep_copy: bool = True):
    assert set(items.keys()).issuperset(
        non_static_names
    ), f"not all names in dict: {non_static_names - set(items.keys())}"

    static_items = {k: v for k, v in items.items() if k not in non_static_names}
    if needs_deep_copy:
        # deep copy used to ensure jax tracks properly through mutation
        static_items = deepcopy(static_items)
    non_static_items = {k: v for k, v in items.items() if k in non_static_names}

    children, non_static_aux_data = tree_flatten(non_static_items)

    return children, dict(
        static_items=static_items,
        non_static_aux_data=non_static_aux_data,
    )


def unflatten_partially_static(aux_data, children):
    assert set(aux_data.keys()) == {"static_items", "non_static_aux_data"}

    static_items = aux_data["static_items"]
    non_static_items = tree_unflatten(aux_data["non_static_aux_data"], children)

    return {**static_items, **non_static_items}


@define
class AttrsPartiallyStatic:
    def non_static_names(self) -> Set[str]:
        return set()

    def needs_deep_copy(self) -> bool:
        return True

    def tree_flatten(self):
        return flatten_partially_static(
            asdict(self, recurse=False),
            non_static_names=self.non_static_names(),
            needs_deep_copy=self.needs_deep_copy(),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**unflatten_partially_static(aux_data, children))


@define
class AttrsPartiallyStaticDefaultNonStatic(AttrsPartiallyStatic):
    def static_names(self) -> Set[str]:
        return set()

    def non_static_names(self) -> Set[str]:
        return set(asdict(self, recurse=False).keys()) - self.static_names()


def default_is_leaf(x):
    return treedef_is_leaf(tree_flatten(x)[1])
