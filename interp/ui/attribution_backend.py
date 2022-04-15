from collections import defaultdict
import dataclasses
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from flax.core.scope import FrozenVariableDict
import jax.numpy as jnp
from attrs import define
import numpy as np

from interp.model.gpt_model import Gpt, get_gpt_block_fn, gpt_call_no_log
from interp.model.model_loading import get_dataset_mean_activations
from interp.tools.attribution_backend_utils import Fake
from interp.tools.interpretability_tools import sequence_tokenize, single_tokenize, toks_to_string_list
import interp.tools.optional as op
from interp.ui.attribution_backend_comp import (
    HeadHiddenSetup,
    LayerType,
    get_starting_unjit,
    block_params,
    get_starting,
    head_hidden_covector,
    StartingOutputs,
    layer_type,
    logits_for_specific_path,
    logits_for_specific_path_stepwise,
    get_outputs,
)
from interp.tools.data_loading import np_log_softmax
from interp.ui.very_named_tensor import VeryNamedTensor


def layer_names(model: Gpt):
    return [f"layer_{i}" for i in range(model.num_layers)]


def head_names(model: Gpt):
    return [str(i) for i in range(model.num_heads)]


def neuron_names(model: Gpt):
    return [str(i) for i in range(model.hidden_size * 4)]


def to_seq_vnt(model: Gpt, tokenizer, tensor, tokens):
    token_strs = [tokenizer.decode([x]) for x in tokens]
    return VeryNamedTensor(
        tensor=tensor,
        title="gradflow",
        units="attribution",
        dim_names=["layer_to", "layer_from", "head_to", "head_from", "seq_to", "seq_from", "qkv_to"],
        dim_types=["layer", "layer", "head", "head", "seq", "seq", "qkv"],
        dim_idx_names=[
            layer_names(model),
            layer_names(model),
            head_names(model),
            head_names(model),
            token_strs,
            token_strs,
            ["q", "k", "v"],
        ],
    )


def to_attribution_vnts(model: Gpt, tokenizer, covectors: Dict[LayerType, jnp.ndarray], tokens, onames, fuse_neurons):
    mlp = covectors.get(
        LayerType.mlps,
        np.zeros((len(onames), model.num_layers, 1 if fuse_neurons else model.hidden_size * 4, len(tokens))),
    )
    token_strs = [tokenizer.decode([x]) for x in tokens]
    return {
        LayerType.embeds.value: VeryNamedTensor(
            tensor=covectors[LayerType.embeds],
            title="gradflow",
            units="attribution",
            dim_names=["output_things", "token_embedding"],
            dim_types=["output_things", "seq"],
            dim_idx_names=[onames, token_strs],
        ),
        LayerType.heads.value: VeryNamedTensor(
            tensor=covectors[LayerType.heads],
            title="gradflow",
            units="attribution",
            dim_names=["output_things", "layer", "head", "seq"],
            dim_types=["output_things", "layer", "head", "seq"],
            dim_idx_names=[onames, layer_names(model), head_names(model), token_strs],
        ),
        LayerType.mlps.value: VeryNamedTensor(
            tensor=mlp,
            title="gradflow",
            units="attribution",
            dim_names=["output_things", "layer", "neuron", "seq"],
            dim_types=["output_things", "layer", "neuron", "seq"],
            dim_idx_names=[onames, layer_names(model), ["o"] if fuse_neurons else neuron_names(model), token_strs],
        ),
    }


AttributionRoot = Dict[str, Any]


def derivs_and_activations_to_attributions(
    derivs_wrt_each_token_qkv: jnp.ndarray, output_activations: Dict[LayerType, jnp.ndarray], num_layers: int
) -> Dict[LayerType, jnp.ndarray]:
    result: Dict[LayerType, jnp.ndarray] = {}
    # print(derivs_wrt_each_token_qkv.shape, [x.shape for x in output_activations.values()])
    for k, v in output_activations.items():
        if k is LayerType.embeds:
            result[k] = jnp.einsum(
                "s d, ... s d -> ... s", v.astype(jnp.float16), derivs_wrt_each_token_qkv.astype(jnp.float16)
            )
        else:
            earlier_attributions = jnp.einsum(
                "l h s d, ... s d -> ... l h s", v.astype(jnp.float16), derivs_wrt_each_token_qkv.astype(jnp.float16)
            )
            pad_shape = list(earlier_attributions.shape)
            pad_shape[-3] = num_layers - earlier_attributions.shape[-3]
            all_layer_attributions = jnp.concatenate([earlier_attributions, jnp.zeros(pad_shape)], axis=-3)
            result[k] = all_layer_attributions
    return result


@dataclasses.dataclass(frozen=True)
class Edge:
    layer_type_to: LayerType
    layer_to: int
    h_to: int
    seq_to: int
    qkv_or_o_to: int
    layer_type_from: LayerType
    layer_from: int
    h_from: int
    seq_from: int

    def get_from(self) -> Dict:
        return dict(
            layer_type_from=self.layer_type_from, layer_from=self.layer_from, h_from=self.h_from, seq_from=self.seq_from
        )

    def get_to(self) -> Dict:
        return dict(
            layer_type_from=self.layer_type_to, layer_from=self.layer_to, h_from=self.h_to, seq_from=self.seq_to
        )


LOGITS_STEPWISE_BY_DEFAULT = True


class AttributionBackend:
    """
    Holds the states of currently ongoing attributions.

    This is a forest where each tree starts at a loss, and each node's children are
    (ideally, but not necessarily, mutually exclusive and collectively exhaustive) inputs
    to the node.
    """

    def __init__(
        self,
        model: Gpt,
        params: FrozenVariableDict,
        tokenizer,
        string: str,
        model_name: str = None,
        model_info: Any = None,
    ):
        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.model_info = model_info

        if self.model_name is not None:
            self.dataset_activation_means = get_dataset_mean_activations(self.model_name)

        self.toks = sequence_tokenize(string)
        self.token_strs = [tokenizer.decode([x]) for x in self.toks]

        model_b = model.bind(params)

        self.pos_embeds = model_b.embedding(jnp.expand_dims(self.toks, 0)).get("pos")
        self.block = get_gpt_block_fn(model, params)

        self.rootKind = None
        self.tree: Optional[Dict[str, Any]] = None
        self.starting_outputs: Optional[StartingOutputs] = None

        self.mask_nodes: Optional[Dict[LayerType, Dict[int, np.ndarray]]] = None
        self.mask_full_covectors: Optional[Dict[LayerType, Dict[int, Any]]] = None

        self.str_to_tok_id = {tokenizer.decode([x]): x for x in range(len(tokenizer))}

    def startTree(
        self,
        attributionRoot: AttributionRoot,
        use_integrated_gradients_loss_log_probs: bool = False,
        subtract_dataset_mean_from_activations: bool = False,
        fuse_neurons: bool = False,
    ):  # TODO support tree wrt whatever head output, not via loss
        """
        Start a new attribution tree to some loss metric.

        Computes and stores the covectors from the loss node(s) to its children.
        """
        if (
            attributionRoot["kind"] == "logprob"
            or attributionRoot["kind"] == "logit"
            or attributionRoot["kind"] == "prob"
        ):
            seq_idx = attributionRoot["data"]["seqIdx"]
            tok_string = attributionRoot["data"]["tokString"]
            comparison_tok_string = attributionRoot["data"].get("comparisonTokString", None)
            print("call startTree", seq_idx, tok_string, comparison_tok_string)
            tok_string = self.str_to_tok_id.get(tok_string, -1)
            comparison_tok_string = self.str_to_tok_id.get(comparison_tok_string, -1)
            self.starting_outputs = get_starting(
                model=self.model,
                params=self.params,
                seqs=jnp.expand_dims(self.toks, 0),
                target=jnp.array([seq_idx, tok_string, comparison_tok_string]),
                is_comparison=comparison_tok_string != -1,
                fake_log_probs="ig" if use_integrated_gradients_loss_log_probs else "none",
                logit_logprob_or_prob=attributionRoot["kind"],
                fuse_neurons=fuse_neurons,
            )
            assert isinstance(self.starting_outputs, StartingOutputs)

            # subtract means from starting
            if subtract_dataset_mean_from_activations:
                if self.model_name:
                    if self.starting_outputs is not None:
                        self.starting_outputs.attn_head.outputs -= np.expand_dims(
                            self.dataset_activation_means["attention"], axis=-2
                        )
                        if self.starting_outputs.mlp is not None:
                            self.starting_outputs.mlp.outputs -= np.expand_dims(
                                self.dataset_activation_means["mlp"], axis=-2
                            )
                else:
                    print("WARNING: DONT HAVE CACHED ACTIVATIONS, NO MODEL NAME")

            self.tree = {
                "idx": {"layerWithIO": self.model.num_layers + 1, "token": seq_idx, "headOrNeuron": 0, "isMlp": False},
                "children": [],
                "covectors": self.starting_outputs.covector[0],
            }
            self.rootKind = attributionRoot["kind"]
            return to_attribution_vnts(
                self.model,
                self.tokenizer,
                {k: np.expand_dims(v, axis=0) for k, v in self.starting_outputs.all_derivs().items()},
                self.toks,
                ["o"],
                fuse_neurons=fuse_neurons,
            )

    def expandTreeNode(
        self,
        tree_path,
        use_integrated_gradients_attn_probs,
        fuse_neurons: bool = True,
        use_half_linear_activation: bool = False,
    ):
        """
        Expand a node in a particular tree.

        The given path is expected to be a connected sequence of nodes in the current forest.
        This method then computes the covectors from the last node to its children. If the last
        node is an attention, it is split into q/k/v before expanding.
        """
        print("call expandTreeNode", tree_path)
        tree = self.tree
        assert tree is not None

        def checkIdxsEqual(a, b):
            return all([a.get(k, np.NaN) == v for k, v in b.items()])

        for node in tree_path[:-1]:
            matching_branches = [i for i, x in enumerate(tree["children"]) if checkIdxsEqual(node, x["idx"])]
            if len(matching_branches) == 0:
                return "no matching branch"
            tree = tree["children"][matching_branches[0]]
            assert tree is not None

        node = tree_path[-1]
        layer_overall = node["layerWithIO"]
        head_or_neuron_idx = node["headOrNeuron"]
        seq_idx = node["token"]
        is_mlp = node["isMlp"]

        if layer_overall == 0:
            return "already at leaf"
        layer_idx = layer_overall - 1
        covectors = tree["covectors"]
        covector = covectors[seq_idx]
        assert self.starting_outputs is not None
        stime = time.time()
        this_covectors = head_hidden_covector(
            self.block,
            block_params(self.params, layer_idx),
            op.unwrap(
                op.unwrap(self.starting_outputs.mlp).inputs if is_mlp else self.starting_outputs.attn_head.inputs
            )[layer_idx],
            self.pos_embeds,
            HeadHiddenSetup(out_seq_idx=seq_idx, target_hidden=covector),
            head_or_neuron_idx,
            fake_attn="ig" if use_integrated_gradients_attn_probs else "none",
            is_attn=not is_mlp,
            fuse_neurons=fuse_neurons,
            fake_mlp_activation="half_linear" if use_half_linear_activation else "none",
        )
        if not is_mlp:
            for qkv in range(3):
                tree["children"].append(
                    {
                        "idx": {**node, "qkv": qkv},
                        "children": [],
                        "covectors": this_covectors[qkv],
                    }
                )
        else:
            tree["children"].append(
                {
                    "idx": node,
                    "children": [],
                    "covectors": this_covectors,
                }
            )
            this_covectors = jnp.expand_dims(this_covectors, axis=0)
        # if len(this_covectors.shape)==2:
        #     this_covectors = jnp.expand_dims(this_covectors,axis=0)
        # if len(this_covectors.shape)==4:
        #     this_covectors = this_covectors[0]
        stime = time.time()
        overall = derivs_and_activations_to_attributions(
            this_covectors,
            self.starting_outputs.earlier_outputs(layer_idx, is_mlp),
            self.model.num_layers,
        )
        return to_attribution_vnts(
            self.model,
            self.tokenizer,
            overall,
            self.toks,
            ["q", "k", "v"] if not is_mlp else ["o"],
            fuse_neurons=fuse_neurons,
        )

    def logitsForSpecificPath(
        self,
        tree_path,
        fake_log_probs: Fake,
        fake_attn: Fake,
        compute_stepwise=LOGITS_STEPWISE_BY_DEFAULT,
        fuse_neurons: bool = False,
    ):
        assert self.starting_outputs is not None

        if len(tree_path) == 0:
            assert self.tree is not None
            out = gpt_call_no_log(self.model, self.params, jnp.expand_dims(self.toks, 0))[0]
            return {"log_probs": np_log_softmax(out[self.tree["idx"]["token"]].reshape(1, -1))[0, :]}

        if compute_stepwise:
            return logits_for_specific_path_stepwise(
                model=self.model,
                params=self.params,
                starting_outputs=self.starting_outputs,
                seqs=jnp.expand_dims(self.toks, 0),
                pos_embeds=self.pos_embeds,
                tree_path=tree_path,
                fake_log_probs=fake_log_probs,
                fake_attn=fake_attn,
                fuse_neurons=fuse_neurons,
                logit_logprob_or_prob=self.rootKind,
            )
        else:
            return logits_for_specific_path(
                self.model,
                self.params,
                jnp.expand_dims(self.toks, 0),
                tree_path,
                fake_log_probs,
                fake_attn,
            )

    def sparseLogitsForSpecificPath(
        self,
        tree_path,
        fake_log_probs: Optional[Fake] = "none",
        fake_attn: Optional[Fake] = "none",
        specific_words=[],
        fuse_neurons: bool = False,
        k=20,
    ):
        specific_tokens = [single_tokenize(t) for t in specific_words]
        non_sparse = self.logitsForSpecificPath(
            tree_path,
            fake_log_probs=op.unwrap_or(fake_log_probs, "none"),
            fake_attn=op.unwrap_or(fake_attn, "none"),
            compute_stepwise=LOGITS_STEPWISE_BY_DEFAULT,
            fuse_neurons=fuse_neurons,
        )
        logits = non_sparse["log_probs"]
        sorted_idxs = jnp.argsort(logits)
        topk = sorted_idxs[-k:]
        topk = topk[::-1]
        bottomk = sorted_idxs[:k]
        return {
            "top": {"values": logits[topk].tolist(), "words": toks_to_string_list(topk)},
            "bottom": {"values": logits[bottomk].tolist(), "words": toks_to_string_list(bottomk)},
            "specific": {"values": logits[np.array(specific_tokens, dtype=np.int32)].tolist(), "words": specific_words},
        }

    def searchAttributionsFromStart(
        self, threshold: float, use_neg=True, use_integrated_gradients_attn_probs=False
    ) -> Tuple[Dict, Dict, Dict]:
        """
        From the current starting outputs, search for nodes whose "total attribution" exceeds a threshold.

        Here the "total attribution" is the sum over all paths via selected nodes. That is, if our model has 3 layers,
        and in the last layer H2.1 and H2.4 have direct attribution exceeding the threshold, they will be selected.
        Then we will check if attribution(H1.1 -> H2.1 -> out) + attribution(H1.1 -> H2.4 -> out)
        + attribution(H1.1 -> out) >= threshold, and if so select H1.1; and so on.

        If use_neg is True, includes nodes whose total attribution is <= -threshold. Both postivie and negative attributions
        are added up in all cases.

        Returns the indices (layer, head/neuron, and seq_idx) and attributions of nodes that get selected.
        """
        assert self.starting_outputs is not None
        assert self.tree is not None
        assert threshold >= 0

        # dict: layer_type -> layer_idx -> [[head_or_neuron if layer_type is not embeds or output, seq_idx], ...]
        self.mask_nodes = {LayerType.embeds: {}, LayerType.heads: {}, LayerType.mlps: {}, LayerType.output: {}}
        # dict: layer_type -> layer_idx -> [array(qkv_or_o, seq, hidden) for node mask_node[layer_type][layer_idx][i] for all i]
        self.mask_full_covectors = {LayerType.heads: {}, LayerType.mlps: {}, LayerType.output: {}}
        # dict: layer_type -> layer_idx -> [attrib to node mask_node[layer_type][layer_idx][i] for all i]
        mask_attribs: Dict[LayerType, dict] = {
            LayerType.embeds: {},
            LayerType.heads: {},
            LayerType.mlps: {},
        }
        masks: Dict[LayerType, np.ndarray] = {
            LayerType.embeds: np.zeros((self.toks.shape[0],), dtype=np.bool8),
            LayerType.heads: np.zeros(
                (self.model.num_layers, self.model.num_heads, self.toks.shape[0]), dtype=np.bool8
            ),
            LayerType.mlps: np.zeros(
                (self.model.num_layers, self.model.hidden_size * 4, self.toks.shape[0]), dtype=np.bool8
            ),
        }

        covectors = self.starting_outputs.covector[0]  # seq, hidden

        self.mask_full_covectors[LayerType.output][self.model.num_layers] = np.expand_dims(
            covectors.clone(), axis=(0, 1)
        )
        self.mask_nodes[LayerType.output][self.model.num_layers] = jnp.array([[self.tree["idx"]["token"]]])

        threshold_mlp = threshold / 1

        def get_and_save_top_nodes(layer, covectors, is_attn):
            assert self.starting_outputs is not None
            layer_out = get_outputs(self.starting_outputs, layer, is_attn)
            print("cov out shapes", covectors.shape, layer_out.shape)
            attribs = jnp.einsum("s d, ... s d -> ... s", covectors, layer_out)  # heads/neurons, seq
            print(attribs.shape)
            threshold_here = threshold_mlp if not is_attn and layer != -1 else threshold
            if use_neg:
                mask = jnp.abs(attribs) >= threshold_here
            else:
                mask = attribs >= threshold_here
            print(layer, is_attn, jnp.var(attribs), jnp.mean(attribs))
            key = layer_type(layer, is_attn, self.model.num_layers)
            nodes = np.nonzero(np.array(mask))  # ([h1, h2, ...], [s1, s2, ...])
            self.mask_nodes[key][layer] = jnp.stack(nodes, axis=1)  # [[h1, s1], [h2, s2], ...]
            mask_attribs[key][layer] = attribs[nodes]  # [a1, a2, ...]
            if layer == -1:
                masks[key] = mask
            else:
                masks[key][layer] = mask
            return self.mask_nodes[key][layer]

        for layer_idx in range(self.model.num_layers - 1, -1, -1):
            print(layer_idx)
            for is_attn in [False, True] if self.model.use_mlp else [True]:
                top_nodes = get_and_save_top_nodes(layer_idx, covectors, is_attn)
                inputs = (
                    op.unwrap(self.starting_outputs.attn_head.inputs)[layer_idx]
                    if is_attn
                    else op.unwrap(op.unwrap(self.starting_outputs.mlp).inputs)[layer_idx]
                )
                full_covectors_list = []
                for [head_or_neuron_idx, seq_idx] in top_nodes:
                    full_covector = head_hidden_covector(
                        block=self.block,
                        block_params=block_params(self.params, layer_idx),
                        inputs=inputs,
                        pos_embeds=self.pos_embeds,
                        setup=HeadHiddenSetup(out_seq_idx=seq_idx, target_hidden=covectors),
                        head_or_neuron_idx=head_or_neuron_idx,
                        fake_attn="ig" if use_integrated_gradients_attn_probs else "none",
                        is_attn=is_attn,
                        get_by_qkv=True,
                    )
                    if not is_attn:  # make all covectors have the same shape; attn ones have an extra qkv dim
                        full_covector = np.expand_dims(full_covector, axis=0)
                    # We are computing the total attribution via all nodes in the mask so far
                    covectors += jnp.sum(full_covector, axis=0)
                    full_covectors_list.append(full_covector)
                self.mask_full_covectors[layer_type(layer_idx, is_attn, self.model.num_layers)][layer_idx] = (
                    None if len(full_covectors_list) == 0 else jnp.stack(full_covectors_list)
                )
        get_and_save_top_nodes(-1, covectors, False)  # embedding
        print("mask counts", {k: jnp.sum(v) for k, v in masks.items()})
        return self.mask_nodes, mask_attribs, masks

    def getAttributionInMask(self, threshold: float, use_neg: bool = True) -> Dict[Edge, float]:
        """
        Get attribution of flow through the current mask for specific edges.

        Returns a tuple of (indices specifying edges, edge attributions) for edges whose attribution
        exceeds the specified threshold.
        """
        assert threshold >= 0

        edge_attribs: Dict[Edge, float] = {}

        def compute_and_save_edge_attribs(this_layer_type: LayerType, layer_idx: int):
            assert self.starting_outputs is not None
            assert self.mask_nodes is not None
            assert self.mask_full_covectors is not None

            covector = self.mask_full_covectors[this_layer_type].get(layer_idx, None)
            if (
                covector is None
            ):  # we only stored covectors for nodes in the mask, and there might not be any for this layer
                return
            attribs = derivs_and_activations_to_attributions(  # get attributions for all earlier layers
                covector,
                self.starting_outputs.earlier_outputs(layer_idx, not is_attn),
                self.model.num_layers,
            )
            print("edge", [x.shape for x in attribs.values()])
            covector_nodes = self.mask_nodes[this_layer_type][layer_idx]
            for from_layer_type, a in attribs.items():
                if use_neg:
                    indices = np.nonzero(jnp.abs(a) >= threshold)
                else:
                    indices = np.nonzero(a >= threshold)
                if not len(indices) or not indices[0].size:  # no edges over threshold found
                    continue
                h_and_seq_tos = covector_nodes[indices[0]]
                for h_and_seq_tos, i in zip(h_and_seq_tos, np.stack(indices, axis=1)):
                    seq_to = h_and_seq_tos[1].item()
                    if this_layer_type == LayerType.output:
                        h_to = 0
                    else:
                        h_to = h_and_seq_tos[0].item()
                    if from_layer_type == LayerType.embeds:
                        [qkv_or_o_to, seq_from] = i[1:].tolist()
                        layer_from = -1
                        h_from = 0
                        if not seq_from in self.mask_nodes[from_layer_type][layer_from]:
                            # the from vertex is not in the mask, discard this edge
                            continue
                    else:
                        [qkv_or_o_to, layer_from, h_from, seq_from] = i[1:].tolist()
                        if not any(np.equal(self.mask_nodes[from_layer_type][layer_from], [h_from, seq_from]).all(1)):
                            # the from vertex is not in the mask, discard this edge
                            continue
                    e = Edge(
                        this_layer_type,
                        layer_idx,
                        h_to,
                        seq_to,
                        qkv_or_o_to,
                        from_layer_type,
                        layer_from,
                        h_from,
                        seq_from,
                    )
                    edge_attribs[e] = a[tuple(i)].item()

        for layer_idx in range(self.model.num_layers - 1, -1, -1):
            print(layer_idx)
            for is_attn in [False, True] if self.model.use_mlp else [True]:
                this_layer_type = layer_type(layer_idx, is_attn, self.model.num_layers)
                compute_and_save_edge_attribs(this_layer_type, layer_idx)

        # Get attributions for edges directly from output
        this_layer_type = LayerType.output
        compute_and_save_edge_attribs(this_layer_type, self.model.num_layers)

        return edge_attribs

    def searchAttributionsFromStartForJS(
        self, threshold: float, edge_threshold: float, neg: bool, use_integrated_gradients_attn_probs=False
    ):
        print("searching set")
        nodes_by_layer_type, attribs_by_layer_type, _ = self.searchAttributionsFromStart(
            threshold, neg, use_integrated_gradients_attn_probs
        )
        jsReturnObj: Dict[str, List[Any]] = {"locations": [], "nodeValues": [], "edges": []}
        for layerType, layers in nodes_by_layer_type.items():
            if layerType == LayerType.output:
                continue
            attrib_by_layer = attribs_by_layer_type[layerType]
            for layer, layer_nodes in layers.items():
                nodes = layer_nodes.tolist()
                attribs = attrib_by_layer[layer].tolist()
                for node, attrib in zip(nodes, attribs):
                    if layerType == LayerType.embeds:
                        jsReturnObj["locations"].append(
                            {"layerWithIO": layer + 1, "token": node[0], "headOrNeuron": 0, "isMlp": False}
                        )
                    else:
                        jsReturnObj["locations"].append(
                            {
                                "layerWithIO": layer + 1,
                                "token": node[1],
                                "headOrNeuron": node[0],
                                "isMlp": layerType == LayerType.mlps,
                            }
                        )
                    jsReturnObj["nodeValues"].append(attrib)
        print("computing lines")
        edges_to_attribs = self.getAttributionInMask(edge_threshold)
        edges = jsReturnObj["edges"]
        for e, val in edges_to_attribs.items():
            locs = {"to": {}, "from": {}, "value": val}
            locs["to"] = {
                "token": e.seq_to,
                "layerWithIO": e.layer_to + 1,
                "headOrNeuron": e.h_to,
                "isMlp": e.layer_type_to == LayerType.mlps,
                "qkv": e.qkv_or_o_to if e.layer_type_to == LayerType.heads else None,
            }
            locs["from"] = {
                "token": e.seq_from,
                "layerWithIO": e.layer_from + 1,
                "headOrNeuron": e.h_from,
                "isMlp": e.layer_type_from == LayerType.mlps,
            }
            print(locs)
            edges.append(locs)
        return jsReturnObj

    def to_dict(self):
        return dict(
            _startTree=self.startTree,
            _expandTreeNode=self.expandTreeNode,
            _logitsForSpecificPath=self.logitsForSpecificPath,
            _sparseLogitsForSpecificPath=self.sparseLogitsForSpecificPath,
            _searchAttributionsFromStart=self.searchAttributionsFromStartForJS,
            _getAttributionInMask=self.getAttributionInMask,
            layerNames=layer_names(self.model),
            headNames=head_names(self.model),
            neuronNames=neuron_names(self.model),
            tokens=[t for t in self.token_strs if t != "[END]"],
            hasMlps=self.model.use_mlp,
            modelName=self.model_name,
            modelInfo=self.model_info,
        )
