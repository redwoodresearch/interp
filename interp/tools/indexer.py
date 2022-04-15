# could be fancyier (if we cared)

from typing import Any
from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node_class
from attrs import frozen

from interp.tools.jax_tree_util import AttrsPartiallyStaticDefaultNonStatic


@register_pytree_node_class
@frozen
class WrapSlice(AttrsPartiallyStaticDefaultNonStatic):
    start: Any
    stop: Any
    step: Any

    @classmethod
    def new(cls, s: slice):
        return cls(start=s.start, stop=s.stop, step=s.step)

    def to_slice(self):
        return slice(self.start, self.stop, self.step)


@register_pytree_node_class
class IdxTup(tuple):
    def tree_flatten(self):
        return tree_flatten(tuple(WrapSlice.new(x) if isinstance(x, slice) else x for x in self))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return IdxTup(x.to_slice() if isinstance(x, WrapSlice) else x for x in tree_unflatten(aux_data, children))


class Indexer:
    """
    Helper for defining slices more easily, also wraps slices so that they are
    valid jax types.
    """

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return IdxTup(idx)
        else:
            return idx


I = Indexer()
