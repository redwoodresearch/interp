from typing import Any, FrozenSet, Optional, Tuple, List, Union

from attrs import frozen
from flax.core.scope import FrozenVariableDict
import numpy as np

from interp.model.gpt_model import Gpt, partial_gpt_call
from interp.tools.assert_never import assert_never
from interp.tools.grad_modify_query import GradModifierConf, ModifierCollectionTreeNode, ModifierCollectionTreeNodeStack
from interp.tools.grad_modify_query_items import ItemIdx, MulConf, StopGradConf, ItemConf
from interp.tools.grad_modify_query_utils import as_op, compose_trees
from interp.tools.interpretability_tools import GetTree, LossesRunnerTreeConfig, losses_runner_tree
from interp.tools.log import Idxs, KeyIdxs
import interp.tools.optional as op


@frozen
class Embeds:
    @property
    def out_name(self):
        return "embedding.overall"


@frozen
class AttnBase:
    def get_by_head(self) -> bool:
        ...

    @property
    def base_name(self):
        return f"blocks.attention"

    @property
    def inp_name(self):
        return f"{self.base_name}.inp"

    @property
    def out_name(self):
        suffix = "out_by_head" if self.get_by_head() else "out"
        return f"{self.base_name}.{suffix}"


@frozen
class Attn(AttnBase):
    by_head: bool = False

    def get_by_head(self) -> bool:
        return self.by_head


class MLPBase:
    def get_by_neuron(self) -> bool:
        ...

    @property
    def base_name(self):
        return f"blocks.mlp"

    @property
    def inp_name(self):
        return f"{self.base_name}.inp"

    @property
    def out_name(self):
        suffix = "out_by_neuron" if self.get_by_neuron() else "out"
        return f"{self.base_name}.{suffix}"


@frozen
class MLP(MLPBase):
    by_neuron: bool = False

    def get_by_neuron(self) -> bool:
        return self.by_neuron


@frozen
class FinalOutput:
    ...


@frozen
class AttnLayer(AttnBase):
    layer: int
    by_head: bool = False

    def get_by_head(self) -> bool:
        return self.by_head


@frozen
class MLPLayer(MLPBase):
    layer: int
    by_neuron: bool = False

    def get_by_neuron(self) -> bool:
        return self.by_neuron


Output = Union[AttnLayer, MLPLayer, Embeds]
InclusiveOutput = Optional[Union[Output, FinalOutput]]


def output_valid(model: Gpt, output: InclusiveOutput):
    return (
        output is None
        or isinstance(output, (Embeds, FinalOutput))
        or (0 <= output.layer < model.num_layers and (isinstance(output, AttnLayer) or model.use_mlp))
    )


def skip_connect(
    model: Gpt, *, start: Output = Embeds(), end: Union[Output, FinalOutput] = FinalOutput()
) -> Optional[ModifierCollectionTreeNode]:
    out: List[ModifierCollectionTreeNode] = []

    assert output_valid(model, start)
    assert output_valid(model, end)

    if isinstance(end, Embeds) or isinstance(start, FinalOutput):
        return None

    start_layer = 0 if isinstance(start, Embeds) else start.layer
    end_layer = model.num_layers if isinstance(end, FinalOutput) else end.layer

    def add(start, end, key):
        if end > start:
            out.append(
                ModifierCollectionTreeNode(StopGradConf(ItemConf(KeyIdxs(key, idxs=Idxs(np.arange(start, end))))))
            )

    attn_skip_start = start_layer + int(isinstance(start, (AttnLayer, MLPLayer)))
    attn_skip_end = end_layer + int(isinstance(end, MLPLayer))

    add(attn_skip_start, attn_skip_end, Attn().inp_name)
    if model.use_mlp:
        mlp_skip_start = start_layer + int(isinstance(start, MLPLayer))
        mlp_skip_end = end_layer

        add(mlp_skip_start, mlp_skip_end, MLP().inp_name)

    if len(out) == 0:
        return None

    return compose_trees(*out)


@frozen
class OutputConf:
    output: Output
    # none allows for all paths, including non-direct
    endpoint: Optional[Union[Output, FinalOutput]] = None
    item_idx: ItemIdx = ItemIdx()
    positive: bool = True
    shape: Tuple[int, ...] = ()
    use_fwd: Optional[bool] = None


def output_tree(model: Gpt, output_conf: OutputConf) -> ModifierCollectionTreeNode:
    output = output_conf.output
    assert output_valid(model, output)
    assert output_valid(model, output_conf.endpoint)

    conf = MulConf(
        ItemConf(
            KeyIdxs(output.out_name) if isinstance(output, Embeds) else KeyIdxs.single(output.out_name, output.layer),
            item_idx=output_conf.item_idx,
            positive=output_conf.positive,
        ),
        shape=output_conf.shape,
    )
    tree = ModifierCollectionTreeNode(conf, use_fwd=output_conf.use_fwd)
    if output_conf.endpoint is not None:
        tree.next_item = skip_connect(model, start=output, end=output_conf.endpoint)

    return tree


def output_trees(model: Gpt, output_confs: List[OutputConf]) -> List[ModifierCollectionTreeNode]:
    return [output_tree(model, conf) for conf in output_confs]


def losses_runner_outputs(
    model: Gpt,
    params: FrozenVariableDict,
    output_confs: List[OutputConf] = [],
    get_extra_tree: Optional[GetTree] = None,
    concat_output_confs: bool = False,
    config: LossesRunnerTreeConfig = LossesRunnerTreeConfig(),
    jit_call: bool = True,
):
    f = partial_gpt_call(model, params, config=Gpt.CallConfig(log_finish=False), jit=jit_call)

    def get_tree(runner, seq_len: int):
        tree: ModifierCollectionTreeNodeStack = op.map(get_extra_tree, lambda get: get(runner, seq_len))

        if len(output_confs) != 0:
            tree = compose_trees(as_op(output_trees(model, output_confs)), tree)

            if concat_output_confs:
                tree = [[p] for p in tree]

        return tree

    return losses_runner_tree(f, get_tree, config)
