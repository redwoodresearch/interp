from enum import Enum
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from flax.core.scope import FrozenVariableDict
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.tree_util import register_pytree_node_class, tree_flatten, tree_unflatten
from attrs import define, asdict, frozen, evolve

from interp.model.gpt_modules import GptBlock
from interp.model.grad_modify_output import MLP, Attn, Embeds
from interp.model.gpt_model import Gpt, get_gpt_block_fn, gpt_call
from interp.tools.assert_never import assert_never
from interp.tools.attribution_backend_utils import (
    Fake,
    fake_to_attn_probs,
    fake_to_mlp_activation,
    fake_to_wrapper,
    stop_qkv_except,
)
from interp.tools.grad_modify_query import (
    ModifierCollectionTreeNodeStack,
    run_queries,
    Query,
    ModifierCollectionTreeNode,
    TargetConf,
    run_query,
)
from interp.tools.grad_modify_query_items import AddConf, ItemConf, ItemIdx, MulConf, StopGradConf
from interp.tools.grad_modify_query_utils import as_op, compose_trees, compose_trees_maybe_empty
from interp.tools.indexer import I
from interp.tools.jax_tree_util import AttrsPartiallyStaticDefaultNonStatic
from interp.tools.log import EnableModSetup, Logger, LoggerCache, KeyIdxs, Idxs, LogInfo, MutLogCache, LogCache
import interp.tools.optional as op


class LayerType(Enum):
    embeds = "embeds"
    mlps = "mlps"
    heads = "heads"
    output = "output"


def layer_type(layer_idx: int, is_attn: bool, num_layers: int) -> LayerType:
    if layer_idx == -1:
        return LayerType.embeds
    if layer_idx == num_layers:
        return LayerType.output
    if is_attn:
        return LayerType.heads
    return LayerType.mlps


@register_pytree_node_class
@define
class StartingItem:
    outputs: jnp.ndarray
    inputs: Optional[jnp.ndarray]
    derivs: jnp.ndarray

    def tree_flatten(self):
        return tree_flatten(asdict(self, recurse=False))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**tree_unflatten(aux_data, children))


@register_pytree_node_class
@define
class StartingOutputs:
    covector: jnp.ndarray
    embeds: StartingItem
    attn_head: StartingItem
    mlp: Optional[StartingItem]

    def all_outputs(self) -> Dict[LayerType, jnp.ndarray]:
        if self.mlp is not None:
            return {
                LayerType.embeds: self.embeds.outputs,
                LayerType.heads: self.attn_head.outputs,
                LayerType.mlps: self.mlp.outputs,
            }
        else:
            return {LayerType.embeds: self.embeds.outputs, LayerType.heads: self.attn_head.outputs}

    def all_derivs(self) -> Dict[LayerType, jnp.ndarray]:
        if self.mlp is not None:
            return {
                LayerType.embeds: self.embeds.derivs,
                LayerType.heads: self.attn_head.derivs,
                LayerType.mlps: self.mlp.derivs,
            }
        else:
            return {LayerType.embeds: self.embeds.derivs, LayerType.heads: self.attn_head.derivs}

    def earlier_outputs(self, layer: int, is_mlp: bool) -> Dict[LayerType, jnp.ndarray]:
        if self.mlp is not None:
            if is_mlp:
                return {
                    LayerType.embeds: self.embeds.outputs,
                    LayerType.heads: self.attn_head.outputs[: layer + 1],
                    LayerType.mlps: self.mlp.outputs[:layer],
                }
            else:
                return {
                    LayerType.embeds: self.embeds.outputs,
                    LayerType.heads: self.attn_head.outputs[:layer],
                    LayerType.mlps: self.mlp.outputs[:layer],
                }
        else:
            return {LayerType.embeds: self.embeds.outputs, LayerType.heads: self.attn_head.outputs[:layer]}

    def tree_flatten(self):
        return tree_flatten(asdict(self, recurse=False))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**tree_unflatten(aux_data, children))


def get_outputs(starting_outputs: StartingOutputs, layer: int, is_attn: bool) -> jnp.ndarray:
    if layer == -1:
        return starting_outputs.embeds.outputs
    elif is_attn:
        return starting_outputs.attn_head.outputs[layer]
    else:
        if starting_outputs.mlp is not None:
            return starting_outputs.mlp.outputs[layer]
        else:
            raise Exception("looking for mlp output, but model doesn't have MLP (or something)")


Covector = jnp.ndarray  # num_parts_to_input_which_is_qkv_for_heads_and_neurons_for_mlp, seq_len, hidden_size


def block_params(params: FrozenVariableDict, layer_idx: int) -> FrozenVariableDict:
    return FrozenVariableDict({"params": params["params"][f"blocks_{layer_idx}"]})


def get_pos_embeds(params: FrozenVariableDict):
    return params["params"]["embedding"]["position_embedding"]["embedding"]


def get_run_model_and_output(
    model: Gpt,
    params: FrozenVariableDict,
    seqs: jnp.ndarray,
    fake_log_probs: Fake,
    logit_logprob_or_prob,
    target: Optional[jnp.ndarray] = None,
    is_comparison: bool = False,
):
    def run(logger: Logger):
        log_info = LogInfo(logger)
        logits, cache = gpt_call(model, params, seqs, log_info=log_info, config=Gpt.CallConfig(log_finish=False))
        orig_type = logits.dtype
        logits = logits.astype(jnp.float32)
        log_softmax = fake_to_wrapper(fake_log_probs)(jax.nn.log_softmax)
        softmax = lambda x: jnp.exp(fake_to_wrapper(fake_log_probs)(jax.nn.log_softmax)(x))
        log_probs = (
            logits
            if logit_logprob_or_prob == "logit"
            else (log_softmax(logits) if logit_logprob_or_prob == "logprob" else softmax(logits))
        )
        log_probs, cache = log_info.log_and_modify(log_probs, "log_probs", cache)
        log_probs = log_probs.astype(orig_type)
        if target is not None:
            loss = log_probs[0, target[0], target[1]]
            if is_comparison:
                loss -= log_probs[0, target[0], target[2]]
            cache = log_info.sub("loss").log(loss, "loss", cache)

        logger.check_finish_cache(cache)

        return cache

    return run


def get_starting_unjit(
    model,
    params,
    seqs,
    target,
    is_comparison,
    fake_log_probs: Fake,
    logit_logprob_or_prob: Literal["logit", "logprob", "prob"],
    fuse_neurons: bool = False,
) -> StartingOutputs:
    """
    Get the initial covectors to loss of the model, which are at the input to the final_out. Also saves the
    inputs and outputs for all attentions and MLPs. This allows for producing the initial
    display and also prepares for future expansions.
    """
    seq_len = seqs.shape[1]

    no_deriv_targets = []

    def add_target(key: str, is_all: bool):
        no_deriv_targets.append(
            TargetConf(KeyIdxs(key, Idxs.all() if is_all else Idxs()), idx=I[:, 0] if is_all else 0)
        )

    add_target(Embeds().out_name, False)
    for x in (Attn(by_head=True),) + ((MLP(by_neuron=not fuse_neurons),) if model.use_mlp else ()):
        add_target(x.out_name, True)
        add_target(x.inp_name, True)

    covec_target = [TargetConf(KeyIdxs("loss.loss"), display_name="covector")]

    vals = run_queries(
        get_run_model_and_output(
            model,
            params,
            seqs,
            fake_log_probs=fake_log_probs,
            logit_logprob_or_prob=logit_logprob_or_prob,
            target=target,
            is_comparison=is_comparison,
        ),
        {
            "output_to_loss": Query(
                targets=covec_target,
                modifier_collection_tree=ModifierCollectionTreeNode(
                    AddConf(ItemConf(KeyIdxs("final_out.inp"), ItemIdx(0)), shape=(seq_len, model.hidden_size))
                ),
            ),
            "no_deriv": Query(targets=no_deriv_targets),
        },
        use_fwd=False,
    )

    covectors = vals["output_to_loss"]["covector"]

    def to_deriv(x):
        return jnp.einsum("s d, ... s d -> ... s", covectors, x)

    def get_item(x: Union[MLP, Attn, Embeds]):
        inputs = None if isinstance(x, Embeds) else vals["no_deriv"][x.inp_name]
        outputs = vals["no_deriv"][x.out_name]
        if fuse_neurons and isinstance(x, MLP):
            outputs = jnp.expand_dims(outputs, axis=-3)
        return StartingItem(outputs=outputs, inputs=inputs, derivs=to_deriv(outputs))

    result = StartingOutputs(
        jnp.expand_dims(covectors, axis=0),
        embeds=get_item(Embeds()),
        attn_head=get_item(Attn(by_head=True)),
        mlp=get_item(MLP(by_neuron=not fuse_neurons)) if model.use_mlp else None,
    )
    return result


get_starting = jax.jit(
    get_starting_unjit,
    static_argnames=[
        "model",
        "is_comparison",
        "fake_log_probs",
        "fuse_neurons",
        "fake_mlp_activation",
        "logit_logprob_or_prob",
    ],
)


@register_pytree_node_class
@frozen
class ForwardHeadHiddenSetup(AttrsPartiallyStaticDefaultNonStatic):
    inp_seq_idx: Union[int, jnp.ndarray]
    input_covectors: jnp.ndarray


@register_pytree_node_class
@frozen
class HeadHiddenSetup(AttrsPartiallyStaticDefaultNonStatic):
    out_seq_idx: Union[int, jnp.ndarray]
    target_hidden: jnp.ndarray


def head_hidden_covector_unjit(
    block: GptBlock,
    block_params,
    inputs: jnp.ndarray,
    pos_embeds: Optional[jnp.ndarray],
    setup: Union[ForwardHeadHiddenSetup, HeadHiddenSetup],  # TODO: better name than setup?
    head_or_neuron_idx,
    fake_attn: Fake,
    is_attn: bool,
    get_by_qkv: bool = True,
    fuse_neurons: bool = True,
    fake_mlp_activation: Fake = "none",
) -> jnp.ndarray:
    """
    Get the derivatives of the input to a layer wrt 'the loss' which
    corresponds to the output dotted by `target_hidden`.
    """
    print("EXPAND FUSE NEURONS", fuse_neurons)
    x: Union[Attn, MLP] = Attn(by_head=True) if is_attn else MLP(by_neuron=not fuse_neurons)

    def loss(logger: Logger):
        assert isinstance(logger, LoggerCache)
        logger = evolve(logger, to_cache=logger.to_cache.union([x.out_name]))

        log = MutLogCache.new(logger)

        block.apply(
            block_params,
            jnp.expand_dims(log.log_and_modify(inputs, "inputs"), 0),
            log=log.sub("blocks").sub("attention" if is_attn else "mlp"),
            **({"pos_embeds": pos_embeds} if is_attn else {}),  # type: ignore
            method=block.attn if is_attn else block.mlp,
        )

        cache = log.cache
        assert isinstance(cache, LogCache)
        out = cache.cache[x.out_name][0]
        if (not fuse_neurons) or is_attn:
            out = out[head_or_neuron_idx]
        print("OUT SHAPE", out.shape)
        if isinstance(setup, ForwardHeadHiddenSetup):
            log.log(out, "loss")
        elif isinstance(setup, HeadHiddenSetup):
            selected_out = out[setup.out_seq_idx]
            log.log(jnp.sum(setup.target_hidden * selected_out), "loss")
        else:
            assert_never(setup)

        log.check_finish()

        return log.cache

    add_inp_idx: Any
    add_multiplier: Union[float, jnp.ndarray]
    add_shape: Tuple[int, ...]
    if isinstance(setup, ForwardHeadHiddenSetup):
        add_inp_idx = setup.inp_seq_idx
        add_multiplier = setup.input_covectors
        add_shape = ()
    elif isinstance(setup, HeadHiddenSetup):
        add_inp_idx = ...
        add_multiplier = 1.0
        add_shape = inputs.shape
    else:
        assert_never(setup)

    return run_query(
        loss,
        Query(
            targets=[TargetConf(KeyIdxs("loss"), display_name="covectors")],
            modifier_collection_tree=compose_trees(
                ModifierCollectionTreeNode(
                    AddConf(
                        ItemConf(KeyIdxs("inputs"), ItemIdx(add_inp_idx, is_static=False)),
                        shape=add_shape,
                        multiplier=add_multiplier,
                    )
                ),
                fake_to_mlp_activation(fake_mlp_activation, idxs=Idxs()) if not is_attn else None,
                fake_to_attn_probs(fake_attn, idxs=Idxs()) if is_attn else None,
                *(
                    [as_op([[compose_trees(*stop_qkv_except(Idxs(), qkv))] for qkv in range(3)])]
                    if is_attn and get_by_qkv
                    else []
                ),
            ),
            use_fwd=isinstance(setup, ForwardHeadHiddenSetup),
        ),
    )["covectors"]


head_hidden_covector = jax.jit(
    head_hidden_covector_unjit,
    static_argnames=["block", "fake_attn", "is_attn", "get_by_qkv", "fuse_neurons", "fake_mlp_activation"],
)


def residual_to_logprobs_unjit(model, params, seqs, covector, seq_idx, fake_log_probs: Fake, logit_logprob_or_prob):
    return run_query(
        get_run_model_and_output(
            model, params, seqs, fake_log_probs=fake_log_probs, logit_logprob_or_prob=logit_logprob_or_prob
        ),
        Query(
            targets=[TargetConf(KeyIdxs("log_probs"), idx=I[0, seq_idx])],
            modifier_collection_tree=compose_trees(
                ModifierCollectionTreeNode(
                    AddConf(
                        ItemConf(KeyIdxs("final_out.inp"), ItemIdx((0, seq_idx), is_static=False)), multiplier=covector
                    )
                ),
            ),
            use_fwd=True,
        ),
    )["log_probs"]


residual_to_logprobs = jax.jit(
    residual_to_logprobs_unjit, static_argnames=["model", "fake_log_probs", "fuse_neurons", "logit_logprob_or_prob"]
)


def logits_for_specific_path_stepwise(
    model: Gpt,
    params: FrozenVariableDict,
    starting_outputs: StartingOutputs,
    seqs: jnp.ndarray,
    pos_embeds,
    tree_path,
    fake_log_probs: Fake,  # TODO actually use this (whether to use ig log softmax)
    fake_attn: Fake,
    fuse_neurons: bool,
    logit_logprob_or_prob,
):
    seq_len = starting_outputs.embeds.outputs.shape[0]

    def bounds_check(layer, head_or_neuron, seq_idx, is_mlp, qkv=None):
        assert -1 <= layer < model.num_layers, layer
        assert 0 <= seq_idx < seq_len, seq_idx
        if is_mlp:
            assert 0 <= head_or_neuron < model.hidden_size * 4, head_or_neuron
            assert qkv == None
        else:
            assert 0 <= head_or_neuron < model.num_heads, head_or_neuron
            if qkv is not None:
                assert 0 <= qkv < 3, qkv

    start_layer_idx = tree_path[-1]["layerWithIO"] - 1
    start_head_or_neuron = tree_path[-1]["headOrNeuron"]
    start_seq_idx = tree_path[-1]["token"]
    start_is_mlp = tree_path[-1]["isMlp"]
    bounds_check(start_layer_idx, start_head_or_neuron, start_seq_idx, start_is_mlp)

    starting_by_is_mlp: Dict[bool, Optional[StartingItem]] = {
        True: op.unwrap_or(starting_outputs.mlp, None),
        False: starting_outputs.attn_head,
    }
    covector = get_outputs(starting_outputs, start_layer_idx, not start_is_mlp)[..., start_seq_idx, :]
    if start_layer_idx != -1:
        covector = covector[start_head_or_neuron]

    prev_seq_idx = start_seq_idx

    for path_node in tree_path[:-1]:
        node_layer_idx = path_node["layerWithIO"] - 1
        node_head_or_neuron = path_node["headOrNeuron"]
        node_seq_idx = path_node["token"]
        node_is_mlp = path_node["isMlp"]
        inp = op.unwrap(op.unwrap(starting_by_is_mlp[node_is_mlp]).inputs)[node_layer_idx]
        if inp.shape[0] == 1 and len(inp.shape) == 3:
            inp = inp[0]
        next_covectors = head_hidden_covector(
            get_gpt_block_fn(model, params),
            block_params(params, node_layer_idx),
            inp,
            pos_embeds=pos_embeds,
            setup=ForwardHeadHiddenSetup(inp_seq_idx=prev_seq_idx, input_covectors=covector),
            head_or_neuron_idx=node_head_or_neuron,
            fake_attn=fake_attn,
            is_attn=not node_is_mlp,
            fuse_neurons=fuse_neurons,
        )
        covector = next_covectors[node_seq_idx]
        if not node_is_mlp:
            covector = covector[:, path_node["qkv"]]

    logprobs = residual_to_logprobs(
        model,
        params,
        seqs,
        covector,
        tree_path[0]["token"],
        fake_log_probs,
        logit_logprob_or_prob=logit_logprob_or_prob,
    )
    return {"log_probs": logprobs}


def logits_for_specific_path_impl_unjit(
    model: Gpt,
    params: FrozenVariableDict,
    seqs: jnp.ndarray,
    start_layer: Union[int, jnp.ndarray],
    start_head_or_neuron: Union[int, jnp.ndarray],
    start_seq_idx: Union[int, jnp.ndarray],
    start_is_mlp: Union[bool, jnp.ndarray],
    last_seq_idx: Union[int, jnp.ndarray],
    head_active_by_layer: jnp.ndarray,
    qkv_idx_by_layer: jnp.ndarray,
    head_idx_by_layer: jnp.ndarray,
    seq_idx_by_layer: jnp.ndarray,
    neuron_idx_by_layer: jnp.ndarray,
    neuron_seq_idx_by_layer: jnp.ndarray,
    neuron_active_by_layer: jnp.ndarray,
    fake_log_probs: Fake,
    fake_attn_probs: Fake,
    static_enable: bool,
    logit_logprob_or_prob,
):
    start_nodes = []

    start_layer_arr: jnp.ndarray = jnp.array(start_layer)
    start_is_embeds = start_layer_arr == -1
    # handle start
    start_nodes.append(
        ModifierCollectionTreeNode(
            MulConf(
                ItemConf(
                    KeyIdxs(Embeds().out_name),
                    ItemIdx(I[:, start_seq_idx], is_static=False),
                    enable_setup=EnableModSetup(enable=start_is_embeds, is_enable_static=static_enable),
                )
            )
        )
    )
    start_is_mlp = jnp.array(start_is_mlp)
    start_nodes.append(
        ModifierCollectionTreeNode(
            MulConf(
                ItemConf(
                    KeyIdxs.single(Attn(by_head=True).out_name, start_layer_arr),
                    ItemIdx(I[:, start_head_or_neuron, start_seq_idx], is_static=False),
                    enable_setup=EnableModSetup(
                        enable=(~start_is_embeds) & (~start_is_mlp), is_enable_static=static_enable
                    ),
                )
            )
        )
    )
    if model.use_mlp:
        start_nodes.append(
            ModifierCollectionTreeNode(
                MulConf(
                    ItemConf(
                        KeyIdxs.single(f"blocks.mlp.linear1", start_layer_arr),
                        ItemIdx(I[:, start_seq_idx, start_head_or_neuron], is_static=False),
                        enable_setup=EnableModSetup(
                            enable=(~start_is_embeds) & start_is_mlp, is_enable_static=static_enable
                        ),
                    )
                )
            )
        )

    # Summing over start nodes like this might be problematic.
    # Like we're hoping jax optimizes away all the ones which aren't enabled.
    # And also hoping this optimization doesn't burn tons of compile time.
    # This could probably be solved by having MulConf take a list of ItemConfs
    # which operate on the same actual alpha value (in fact I've already
    # written the (minimal) code needed for this!). I won't change this yet
    # because it might not be needed and adds complexity
    nodes_to_compose: List[ModifierCollectionTreeNodeStack] = [as_op(start_nodes)]

    def check_size(x: Union[Tuple[Any, ...], jnp.ndarray]):
        if isinstance(x, tuple):
            assert len(x) == model.num_layers
        else:
            assert x.shape == (model.num_layers,)

    check_size(head_active_by_layer)
    check_size(neuron_active_by_layer)
    check_size(head_idx_by_layer)
    check_size(qkv_idx_by_layer)
    check_size(seq_idx_by_layer)

    def block_stop_grad(key: str, enable: Optional[jnp.ndarray] = None, item_idx: ItemIdx = ItemIdx()):
        return ModifierCollectionTreeNode(
            StopGradConf(
                ItemConf(
                    KeyIdxs(key, Idxs.all()),
                    item_idx=item_idx,
                    enable_setup=op.unwrap_or(
                        op.map(
                            enable, lambda enable: EnableModSetup(enable, is_enable_static=False, enable_by_idx=True)
                        ),
                        EnableModSetup(),
                    ),
                ),
            )
        )

    nodes_to_compose.extend(
        [
            # fine to leave these unconditionally enabled
            *stop_qkv_except(Idxs.all(), qkv_idx_by_layer, static=False),
            block_stop_grad(
                Attn(by_head=True).out_name,
                item_idx=ItemIdx(
                    (head_idx_by_layer, seq_idx_by_layer),
                    get_idx_or_mask=lambda vals, i: I[:, vals[0][i], vals[1][i]],
                    is_static=False,
                    except_idx=True,
                ),
            ),
            # either res stopped or head output stoppped
            block_stop_grad("blocks.inp_res", enable=head_active_by_layer | neuron_active_by_layer),
            block_stop_grad(Attn().out_name, enable=~head_active_by_layer),
        ]
    )

    if model.use_mlp:
        nodes_to_compose.extend(
            [
                block_stop_grad(
                    "blocks.mlp.linear1",
                    item_idx=ItemIdx(
                        (neuron_seq_idx_by_layer, neuron_idx_by_layer),
                        get_idx_or_mask=lambda vals, i: I[:, vals[0][i], vals[1][i]],
                        is_static=False,
                        except_idx=True,
                    ),
                ),
                block_stop_grad(
                    "blocks.inp_res_for_mlp",
                    enable=head_active_by_layer & neuron_active_by_layer,
                ),
                block_stop_grad(MLP().out_name, enable=~neuron_active_by_layer),
            ]
        )

    nodes_to_compose.extend(op.it(fake_to_attn_probs(fake_attn_probs)))

    tree = compose_trees_maybe_empty(*nodes_to_compose)

    return run_query(
        get_run_model_and_output(
            model, params, seqs, fake_log_probs=fake_log_probs, logit_logprob_or_prob=logit_logprob_or_prob
        ),
        Query(
            targets=[
                TargetConf(KeyIdxs("final_out.logits"), "outputs", idx=(0, last_seq_idx)),
                TargetConf(KeyIdxs("log_probs"), "log_probs", idx=(0, last_seq_idx)),
            ],
            modifier_collection_tree=tree,
        ),
    )


logits_for_specific_path_impl = jax.jit(
    partial(logits_for_specific_path_impl_unjit, static_enable=False),
    static_argnames=["model", "fake_log_probs", "fake_attn_probs", "logit_logprob_or_prob"],
)


# TODO: should be wrapped for to_dict
def logits_for_specific_path(
    model: Gpt, params: FrozenVariableDict, seqs: jnp.ndarray, tree_path, fake_log_probs: Fake, fake_attn: Fake
):
    """
    Wrapper for impl which converts to array format for path.
    """

    seq_len = seqs.shape[1]

    def bounds_check(layer, head_or_neuron, seq_idx, is_mlp, qkv=None):
        assert -1 <= layer < model.num_layers, layer
        assert 0 <= seq_idx < seq_len, seq_idx
        if is_mlp:
            assert 0 <= head_or_neuron < model.hidden_size * 4, head_or_neuron
            assert qkv == None
        else:
            assert 0 <= head_or_neuron < model.num_heads, head_or_neuron
            if qkv is not None:
                assert 0 <= qkv < 3, qkv

    start_layer_idx = tree_path[-1]["layerWithIO"] - 1
    start_head_or_neuron = tree_path[-1]["headOrNeuron"]
    start_seq_idx = tree_path[-1]["token"]
    start_is_mlp = tree_path[-1]["isMlp"]
    bounds_check(start_layer_idx, start_head_or_neuron, start_seq_idx, start_is_mlp)

    handled_layers = set()

    head_active_by_layer: jnp.ndarray = jnp.full(model.num_layers, False)
    qkv_idx_by_layer: jnp.ndarray = jnp.zeros(model.num_layers, dtype=jnp.int32)
    head_idx_by_layer: jnp.ndarray = jnp.zeros(model.num_layers, dtype=jnp.int32)
    head_seq_idx_by_layer: jnp.ndarray = jnp.zeros(model.num_layers, dtype=jnp.int32)

    neuron_active_by_layer: jnp.ndarray = jnp.full(model.num_layers, False)
    neuron_idx_by_layer: jnp.ndarray = jnp.zeros(model.num_layers, dtype=jnp.int32)
    neuron_seq_idx_by_layer: jnp.ndarray = jnp.zeros(model.num_layers, dtype=jnp.int32)

    last_seq_idx = tree_path[0]["token"]

    attn_path = tree_path if start_layer_idx >= 0 else tree_path[:-1]

    for attrib_loc in attn_path:
        layer_idx = attrib_loc["layerWithIO"] - 1
        seq_idx = attrib_loc["token"]
        head_or_neuron = attrib_loc["headOrNeuron"]
        assert layer_idx >= 0
        is_mlp = attrib_loc["isMlp"]
        strang = f"{layer_idx}{is_mlp}"
        assert strang not in handled_layers, "repeated node"
        handled_layers.add(strang)
        if is_mlp:
            neuron_active_by_layer = neuron_active_by_layer.at[layer_idx].set(True)
            neuron_idx_by_layer = neuron_idx_by_layer.at[layer_idx].set(head_or_neuron)
            neuron_seq_idx_by_layer = neuron_seq_idx_by_layer.at[layer_idx].set(seq_idx)
        else:
            qkv = [attrib_loc["qkv"]] if "qkv" in attrib_loc else []
            bounds_check(layer_idx, head_or_neuron, seq_idx, is_mlp, *qkv)

            head_active_by_layer = head_active_by_layer.at[layer_idx].set(True)
            if len(qkv) > 0:
                qkv_idx_by_layer = qkv_idx_by_layer.at[layer_idx].set(*qkv)
            head_idx_by_layer = head_idx_by_layer.at[layer_idx].set(head_or_neuron)
            head_seq_idx_by_layer = head_seq_idx_by_layer.at[layer_idx].set(seq_idx)

    return logits_for_specific_path_impl(
        model=model,
        params=params,
        seqs=seqs,
        start_layer=start_layer_idx,
        start_head_or_neuron=start_head_or_neuron,
        start_seq_idx=start_seq_idx,
        start_is_mlp=start_is_mlp,
        last_seq_idx=last_seq_idx,
        head_active_by_layer=head_active_by_layer,
        qkv_idx_by_layer=qkv_idx_by_layer,
        head_idx_by_layer=head_idx_by_layer,
        seq_idx_by_layer=head_seq_idx_by_layer,
        neuron_idx_by_layer=neuron_idx_by_layer,
        neuron_seq_idx_by_layer=neuron_seq_idx_by_layer,
        neuron_active_by_layer=neuron_active_by_layer,
        fake_log_probs=fake_log_probs,
        fake_attn_probs=fake_attn,
    )
