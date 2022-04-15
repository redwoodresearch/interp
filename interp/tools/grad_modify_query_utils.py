from __future__ import annotations

from typing import Protocol, Optional, Tuple, TypeVar, List, Any, overload

from attrs import evolve, frozen

from interp.tools.assert_never import assert_never
from interp.tools.grad_modify_query import (
    ModifierCollectionTreeNode,
    ModifierCollectionTreeNodeStack,
    Query,
    TargetConf,
    run_query,
)
from interp.tools.grad_modify_query_items import AddConf, ItemConf, ItemIdx, StopGradConf
from interp.tools.log import KeyIdxs


class TreeTraverseAccum(Protocol):
    def add(self, x: ModifierCollectionTreeNode) -> None:
        return

    def branch(self) -> TreeTraverseAccum:
        ...

    def finish(self) -> ModifierCollectionTreeNodeStack:
        ...


def update_tree(node: ModifierCollectionTreeNodeStack, accum: TreeTraverseAccum, branching_allowed=True):
    if node is None:
        return accum.finish()
    elif isinstance(node, ModifierCollectionTreeNode):
        accum.add(node)
        return evolve(node, next_item=update_tree(node.next_item, accum))
    elif isinstance(node, list):
        assert branching_allowed
        return [update_tree(p, accum.branch()) for p in node]
    else:
        assert_never(node)


@overload
def as_op(x: Optional[ModifierCollectionTreeNode]) -> Optional[ModifierCollectionTreeNode]:
    """
    for avoiding typing related sins
    (also known as the identity function)
    """
    ...


@overload
def as_op(x: List[ModifierCollectionTreeNode]) -> List[Optional[ModifierCollectionTreeNode]]:
    ...


@overload
def as_op(x: List[List[ModifierCollectionTreeNode]]) -> List[List[Optional[ModifierCollectionTreeNode]]]:
    ...


@overload
def as_op(x: List[Optional[ModifierCollectionTreeNode]]) -> List[Optional[ModifierCollectionTreeNode]]:
    ...


@overload
def as_op(x: List[List[Optional[ModifierCollectionTreeNode]]]) -> List[List[Optional[ModifierCollectionTreeNode]]]:
    ...


def as_op(x):
    # it was optional all along
    return x


TreeVar = TypeVar(
    "TreeVar",
    ModifierCollectionTreeNode,
    List[ModifierCollectionTreeNode],
    List[List[ModifierCollectionTreeNode]],
    List[Optional[ModifierCollectionTreeNode]],
    List[List[Optional[ModifierCollectionTreeNode]]],
)


# Place trees end to end. If branching is allowed, total number of final paths
# will be the product of the number of paths in each input.
def compose_trees(
    first_tree: TreeVar, *trees: ModifierCollectionTreeNodeStack, branching_allowed: bool = True
) -> TreeVar:
    inp: ModifierCollectionTreeNodeStack = first_tree  # type: ignore[assignment]
    out = compose_trees_maybe_empty(inp, *trees, branching_allowed=branching_allowed)
    return out  # type: ignore[return-value]


# allows for tree to be empty
def compose_trees_maybe_empty(
    *trees: ModifierCollectionTreeNodeStack, branching_allowed: bool = True
) -> ModifierCollectionTreeNodeStack:
    if len(trees) == 0:
        return None
    else:
        first = trees[0]
        rest = trees[1:]

        class CombineAccum(TreeTraverseAccum):
            def branch(self) -> TreeTraverseAccum:
                return self

            def finish(self) -> ModifierCollectionTreeNodeStack:
                return compose_trees_maybe_empty(*rest, branching_allowed=branching_allowed)

        return update_tree(first, CombineAccum(), branching_allowed=branching_allowed)


@frozen
class DerivOpBuilder:
    loggable: Any
    queries: List[Query]
    key_idxs: KeyIdxs
    op: Any

    def get(self):
        return AddConf(
            ItemConf(self.key_idxs), multiplier=self.op(*(run_query(self.loggable, query) for query in self.queries))
        )


class MulBuilder:
    def __init__(
        self,
        loggable: Any,
        tree_l: ModifierCollectionTreeNodeStack,
        tree_r: ModifierCollectionTreeNodeStack,
        mul_l_key_idxs: KeyIdxs,
        mul_r_key_idxs: KeyIdxs,
        mul_op: Any,
        key_idxs: KeyIdxs,
        mulled_item_idx: ItemIdx = ItemIdx(),
        shape: Tuple[int, ...] = (),
        idx_l=...,
        idx_r=...,
        use_fwd_l: bool = True,
        use_fwd_r: bool = True,
        allow_non_first_deriv_l: bool = False,
        allow_non_first_deriv_r: bool = False,
    ) -> None:
        assert not allow_non_first_deriv_l and not allow_non_first_deriv_r, "unsupported (requires stop grad per alpha)"

        self.query_l = Query(
            targets=[TargetConf(mul_l_key_idxs, "l", idx=idx_l)],
            modifier_collection_tree=tree_l,
            use_fwd=use_fwd_l,
            allow_non_first_deriv=allow_non_first_deriv_l,
        )
        self.query_r = Query(
            targets=[TargetConf(mul_r_key_idxs, "r", idx=idx_r)],
            modifier_collection_tree=tree_r,
            use_fwd=use_fwd_r,
            allow_non_first_deriv=allow_non_first_deriv_r,
        )

        self.op_builder = DerivOpBuilder(
            loggable, [self.query_l, self.query_r], key_idxs, lambda l, r: mul_op(l["l"], r["r"])
        )
        self.mul_l_key_idxs = mul_l_key_idxs
        self.mul_r_key_idxs = mul_r_key_idxs
        self.mulled_item_idx = mulled_item_idx
        self.shape = shape

    def conjunctive(self) -> ModifierCollectionTreeNode:
        conf = self.op_builder.get()
        return ModifierCollectionTreeNode(
            evolve(conf, conf=evolve(conf.conf, item_idx=self.mulled_item_idx), shape=self.shape)
        )

    def remove_conjunctive_from_sum(self, remove_conjunctive: bool = True) -> List[ModifierCollectionTreeNode]:
        out = [
            compose_trees(
                ModifierCollectionTreeNode(StopGradConf(ItemConf(self.mul_r_key_idxs))),
                self.query_l.modifier_collection_tree,
            ),
            compose_trees(
                ModifierCollectionTreeNode(StopGradConf(ItemConf(self.mul_l_key_idxs))),
                self.query_r.modifier_collection_tree,
            ),
        ]

        if remove_conjunctive:
            out.append(evolve(self.conjunctive(), positive=False))

        return out
