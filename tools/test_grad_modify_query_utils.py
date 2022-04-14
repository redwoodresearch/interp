from functools import partial
from typing import Dict, Optional
from jax.interpreters.ad import JVPTracer

import pytest
from attrs import frozen

from interp.tools.grad_modify_query import (
    GradModifierConf,
    ModifierCollectionTreeNode as MCTN,
    ModifierCollectionTreeNodeStack as MCTNStack,
)
from interp.tools.grad_modify_query_utils import compose_trees as ct, compose_trees_maybe_empty


@frozen
class ConfForTest(GradModifierConf):
    i: int = 0
    has_val: bool = False
    positive: bool = True

    def get_mod(self, _my_val: JVPTracer, _all_vals: Dict[GradModifierConf, JVPTracer]) -> None:
        return None

    def get_val(self) -> Optional[float]:
        if self.has_val:
            return 1.0
        return None


ns = [ConfForTest(i=i) for i in range(100)]
vs = [ConfForTest(i=i, has_val=True) for i in range(100)]


def test_combine_trees_no_branch():
    cpnb = partial(compose_trees_maybe_empty, branching_allowed=False)
    assert None == cpnb()
    assert None == cpnb(None)

    assert ns[0] != ns[1]  # equality is id

    assert MCTN(ns[0]) == cpnb(MCTN(ns[0]))
    assert MCTN(ns[0], next_item=MCTN(ns[1])) == cpnb(MCTN(ns[0]), MCTN(ns[1]))
    assert MCTN(ns[0], next_item=MCTN(ns[1], next_item=MCTN(ns[2], next_item=MCTN(ns[3])))) == cpnb(
        MCTN(ns[0]), MCTN(ns[1]), MCTN(ns[2]), MCTN(ns[3])
    )
    assert MCTN(ns[0], next_item=MCTN(ns[1], next_item=MCTN(ns[2], next_item=MCTN(ns[3])))) == cpnb(
        MCTN(ns[0]), MCTN(ns[1]), MCTN(ns[2]), MCTN(ns[3]), None
    )
    assert MCTN(ns[0], next_item=MCTN(ns[1], next_item=MCTN(ns[2], next_item=MCTN(ns[3])))) == cpnb(
        MCTN(ns[0]), MCTN(ns[1]), MCTN(ns[2]), MCTN(ns[3]), None, None, None
    )
    assert MCTN(
        ns[0], next_item=MCTN(ns[1], next_item=MCTN(ns[2], next_item=MCTN(ns[3], next_item=MCTN(ns[4]))))
    ) == cpnb(MCTN(ns[0]), MCTN(ns[1]), MCTN(ns[2]), MCTN(ns[3]), None, None, None, MCTN(ns[4]))

    with pytest.raises(AssertionError):
        cpnb([None])
    with pytest.raises(AssertionError):
        cpnb([None, None])
    with pytest.raises(AssertionError):
        cpnb([MCTN(ns[0])])
    with pytest.raises(AssertionError):
        cpnb([MCTN(ns[0]), None])
    with pytest.raises(AssertionError):
        cpnb([MCTN(ns[0]), MCTN(ns[1])])
    with pytest.raises(AssertionError):
        cpnb([[MCTN(ns[0])]])


def test_combine_trees_branching():
    assert [
        MCTN(ns[0], next_item=MCTN(ns[3])),
        MCTN(ns[1], next_item=MCTN(ns[3])),
        MCTN(ns[2], next_item=MCTN(ns[3])),
    ] == ct([MCTN(ns[0]), MCTN(ns[1]), MCTN(ns[2])], MCTN(ns[3]))
    assert MCTN(ns[0], next_item=[MCTN(ns[1]), MCTN(ns[2]), MCTN(ns[3])]) == ct(
        MCTN(ns[0]), [MCTN(ns[1]), MCTN(ns[2]), MCTN(ns[3])]
    )
    sub_1: MCTNStack = [MCTN(ns[9]), MCTN(ns[10]), MCTN(ns[11])]
    sub_2: MCTNStack = [[MCTN(ns[6], next_item=sub_1), MCTN(ns[7], next_item=sub_1)], [MCTN(ns[8], next_item=sub_1)]]
    sub_3: MCTNStack = MCTN(ns[5], next_item=sub_2)

    expected = MCTN(
        ns[0],
        next_item=[
            MCTN(ns[1], next_item=sub_3),
            MCTN(ns[2], next_item=sub_3),
            MCTN(ns[3], next_item=MCTN(ns[4], next_item=sub_3)),
        ],
    )
    actual = ct(
        MCTN(ns[0]),
        [MCTN(ns[1]), MCTN(ns[2]), MCTN(ns[3], next_item=MCTN(ns[4]))],
        MCTN(ns[5]),
        [[MCTN(ns[6]), MCTN(ns[7])], [MCTN(ns[8])]],
        [MCTN(ns[9]), MCTN(ns[10]), MCTN(ns[11])],
    )

    expected.next_item_list_node_check()[0].next_item_node_check().term() == actual.next_item_list_node_check()[
        0
    ].next_item_node_check().term()
    assert expected == actual


# TODO: restore test_mul_builder from previous commit (see blame on this line)

if __name__ == "__main__":
    test_combine_trees_branching()
