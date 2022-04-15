from functools import partial
import itertools

import numpy as np
from interp.model.gpt_model import get_gpt_block_fn, inference
import jax
import jax.numpy as jnp
from tqdm import tqdm
from interp.model.grad_modify_fake import ablation_softmax_probs
from interp.model.model_loading import get_dataset_mean_activations

from interp.tools.indexer import I
from interp.tools.grad_modify_query import (
    ModifierCollectionTreeNode,
    ModifierCollectionTreeNodeStack,
    Query,
    TargetConf,
    run_query,
)
from interp.tools.grad_modify_query_items import ItemConf, MulConf, NoneConf, StopGradConf
from interp.tools.grad_modify_query_utils import compose_trees
from interp.tools.interpretability_tools import get_interp_tokenizer
from interp.model.gpt_model import logprobs_on_correct, loss_by_token_loggable
from interp.tools.log import KeyIdxs, LoggerCache
from interp.ui.attribution_backend_comp import block_params, get_pos_embeds
from interp.tools.data_loading import get_val_seqs
from interp.ui.very_named_tensor import VeryNamedTensor, vnt_guessing_shit_model_tokens
import interp.cui as cui

batch_size = 1
seq_len = 128


def diff_between_model_losses_unjit(model1, params1, model2, params2, input_ids):
    logprobs1 = logprobs_on_correct(inference(model1, params1, input_ids, jit=False)["final_out.logits"], input_ids)
    logprobs2 = logprobs_on_correct(inference(model2, params2, input_ids, jit=False)["final_out.logits"], input_ids)
    diff = logprobs1 - logprobs2
    return diff


diff_between_model_losses = jax.jit(diff_between_model_losses_unjit, static_argnames=["model1", "model2"])


def between_mask_ablation_unjit(model, params, input_ids, head_mask, mlp_mask, model_name):
    dataset_activation_means = get_dataset_mean_activations(model_name)

    reference_log = LoggerCache(
        to_cache=set([f"blocks.{l}.attention.inp" for l in range(model.num_layers)] + ["loss.loss_by_token"])
    )
    loss_by_token_loggable(model, params, input_ids)(reference_log)
    normal_residual_stream_layers = jnp.stack(
        [reference_log[f"blocks.{l}.attention.inp"] for l in range(model.num_layers)], axis=0
    )
    in_mask_residual_stream = jnp.zeros_like(normal_residual_stream_layers[0])
    overall_residual_stream = normal_residual_stream_layers[0]
    block = get_gpt_block_fn(model, params)
    pos_embeds = get_pos_embeds(params)

    for layer in range(model.num_layers):
        layer_mlp_mask = mlp_mask[layer]
        layer_head_mask = head_mask[layer]

        layer_block_params = block_params(params, layer)
        log = LoggerCache(to_cache=set([".attention.out_by_head", ".mlp.out_by_neuron"]))
        overall_residual_stream = block.apply(
            layer_block_params,
            overall_residual_stream - in_mask_residual_stream,
            log,
            "",
            pos_embeds,
        )[0]
        in_mask_residual_stream += jnp.sum(log[".attention.out_by_head"] * layer_head_mask[None, :, None, None], axis=1)
        in_mask_residual_stream -= jnp.sum(
            dataset_activation_means["attention"][layer] * layer_head_mask[:, None], axis=0
        )
        in_mask_residual_stream += jnp.sum(log[".mlp.out_by_neuron"] * layer_mlp_mask[None, :, None, None], axis=1)
        in_mask_residual_stream -= jnp.sum(dataset_activation_means["mlp"][layer] * layer_mlp_mask[:, None], axis=0)

    logits = model.bind(params).out(overall_residual_stream, LoggerCache(), "")
    ablated_logprobs = logprobs_on_correct(logits, input_ids)
    reference_logprobs_on_correct = reference_log["loss.loss_by_token"]
    logprob_difference = reference_logprobs_on_correct - ablated_logprobs
    return logprob_difference


between_mask_ablation = jax.jit(between_mask_ablation_unjit, static_argnames=["model", "model_name"])


def embeds_through_mask_ablation_unjit(model, params, input_ids, embed_mask, head_mask, mlp_mask):
    seq_len = input_ids.shape[1]

    result = run_query(
        loss_by_token_loggable(model, params, input_ids),
        query=Query(
            targets=[TargetConf(KeyIdxs("loss.loss_by_token"), display_name="loss")],
            modifier_collection_tree=compose_trees(
                ModifierCollectionTreeNode(MulConf(ItemConf("embedding.tok"))),
                ModifierCollectionTreeNode(
                    StopGradConf(ItemConf("embedding.tok"), mask=embed_mask[None, :, None], except_idx=True)
                ),
                *list(
                    itertools.chain(
                        *[
                            [
                                ablation_softmax_probs(layer),
                                ModifierCollectionTreeNode(
                                    StopGradConf(
                                        ItemConf(KeyIdxs(f"blocks.{layer}.attention.out_by_head")),
                                        mask=head_mask[layer][None, :, :, None],
                                        except_idx=True,
                                    )
                                ),
                                # ModifierCollectionTreeNode(
                                # StopGradConf(
                                #     ItemConf(KeyIdxs(f"blocks.{layer}.mlp.linear1")),
                                #     mask=jnp.transpose(mlp_mask[layer],(1,0))[None, :, :],
                                #     except_idx=True,
                                # ),
                                # )
                            ]
                            for layer in range(model.num_layers)
                        ]
                    ),
                ),
            ),
            use_fwd=True,
        ),
    )["loss"]
    print("RESULT SHAPE", result.shape)
    return result


embeds_through_mask_ablation = jax.jit(embeds_through_mask_ablation_unjit, static_argnames=["model"])


def get_max_path_contribution_batch_unjit(model, params, path, input_ids):

    result = run_query(
        loss_by_token_loggable(model, params, input_ids),
        query=Query(
            targets=[TargetConf(KeyIdxs("loss.loss_by_token"), display_name="loss")],
            modifier_collection_tree=compose_trees(
                ModifierCollectionTreeNode(MulConf(ItemConf("blocks.0.attention.inp"), shape=I[1, seq_len, 1])),
                *[
                    ModifierCollectionTreeNode(
                        StopGradConf(
                            ItemConf(
                                KeyIdxs(f"blocks.{path_node[0]}.attention.out_by_head"), idx=I[:, path_node[1], :, :]
                            ),
                            except_idx=True,
                        )
                    )
                    for path_node in path
                ],
            ),
            use_fwd=True,
        ),
    )["loss"]
    print("RESULT SHAPE", result.shape)
    result = jnp.transpose(result[:, :, 0, :, 0], (0, 1, 2))
    return result


get_max_path_contribution_batch = jax.jit(get_max_path_contribution_batch_unjit, static_argnames=["model", "path"])


def get_mlp_stats_unjit(model, params, input_ids):

    result = run_query(
        partial(model.apply, params, input_ids),
        query=Query(
            targets=[TargetConf(f"blocks.{l}.mlp.linear1", display_name=f"{l}") for l in range(model.num_layers)],
            use_fwd=True,
        ),
    )
    result = jnp.stack([result[str(i)] for i in range(model.num_layers)], axis=1)
    print("RESULT SHAPE", result.shape)
    return result


get_mlp_stats = jax.jit(get_mlp_stats_unjit, static_argnames=["model"])


def get_mlp_attribs_unjit(model, params, layer, input_ids):

    result = run_query(
        loss_by_token_loggable(model, params, input_ids),
        query=Query(
            targets=[TargetConf(KeyIdxs("loss.loss_by_token"), display_name="loss")],
            modifier_collection_tree=ModifierCollectionTreeNode(
                MulConf(ItemConf(KeyIdxs(f"blocks.{layer}.mlp.linear1")), shape=[1, 1, model.hidden_size * 4])
            ),
            use_fwd=False,
        ),
    )["loss"][:, :, 0, 0]
    return result


get_mlp_attribs = jax.jit(get_mlp_attribs_unjit, static_argnames=["model", "layer"])


def get_mlp_jac_unjit(model, params, input_ids):

    result = run_query(
        partial(model.apply, params, input_ids),
        query=Query(
            targets=[TargetConf(KeyIdxs("blocks.mlp.out"), display_name="loss")],
            modifier_collection_tree=ModifierCollectionTreeNode(
                MulConf(ItemConf(KeyIdxs(f"blocks.mlp.inp")), shape=[1, 1, model.hidden_size])
            ),
            use_fwd=True,
        ),
    )["loss"][
        :, :, :, 0, 0, :
    ]  # batch seq out in
    return result


get_mlp_jac = jax.jit(get_mlp_jac_unjit, static_argnames=["model", "layer"])


def get_max_path_contribution(model, params, path, n_seqs=1000):
    dataset_here = get_val_seqs()[: n_seqs // batch_size * batch_size, :seq_len]
    dataset_batches = dataset_here.reshape(-1, batch_size, seq_len)
    result = []
    for batch in tqdm(dataset_batches):
        tensor = get_max_path_contribution_batch(model, params, path, batch)
        tensor = np.array(tensor, dtype=np.float16)
        result.append(tensor)
    result = np.concatenate(result, axis=0)
    return result, dataset_here


def tvnt(tensor, token_ids, name="attribution"):
    top_strs = [get_interp_tokenizer().decode([x]) for x in token_ids]
    return VeryNamedTensor(
        tensor.astype(np.float32),
        dim_names=["to", "from"],
        dim_types=["seq", "seq"],
        dim_idx_names=[top_strs, top_strs],
        units="attribution",
        title=name,
    )


def show_histogram_ascii(hist_weights, hist_edges):
    print("hi")


def run_fn_on_dataset(fn, n_seqs=1000, batch_size=4, seq_len=214):
    dataset_here = get_val_seqs()[: n_seqs // batch_size * batch_size, :seq_len]
    dataset_batches = dataset_here.reshape(-1, batch_size, seq_len)
    result = []
    for batch in tqdm(dataset_batches):
        tensor = fn(batch)
        tensor = np.array(tensor, dtype=np.float16)
        result.append(tensor)
    result = np.concatenate(result, axis=0)
    return result, dataset_here


def show_top_docs_fn(fn, n_seqs=1000, batch_size=4, seq_len=214, model=None):
    result, token_ids = run_fn_on_dataset(fn, n_seqs=n_seqs, batch_size=batch_size, seq_len=seq_len)
    hist_weights, hist_edges = np.histogram(result)
    show_histogram_ascii(hist_weights, hist_edges)
    docs_fn_max = result
    while len(docs_fn_max.shape) > 1:
        docs_fn_max = np.amax(docs_fn_max, axis=-1)
    ids_by_max = np.argsort(docs_fn_max)
    ids_to_show = [
        0,
        1,
        2,
        -1,
        -2,
        -3,
        ids_by_max.shape[0] // 2,
        ids_by_max.shape[0] // 2 + 1,
        ids_by_max.shape[0] // 2 + 2,
    ]
    return cui.show_tensors(
        *[
            vnt_guessing_shit_model_tokens(
                result[i], model=model, tokens=token_ids[i], title=f"fn sorted index {i}"
            ).to_lvnt()
            for i in ids_to_show
        ],
        name="attribution_search",
    )


def show_path_contribution_histogram(model, params, path, n_seqs=1000):
    result, token_ids = get_max_path_contribution(model, params, path, n_seqs)
    hist_weights, hist_edges = np.histogram(result)
    show_histogram_ascii(hist_weights, hist_edges)
    ids_by_max = np.argsort(np.amax(np.amax(result, axis=-1), axis=-1))
    ids_to_show = [0, 1, 2, -1, -2, -3, ids_by_max.shape[0] // 2]
    return cui.show_tensors(
        *[tvnt(result[i], token_ids[i], name=f"{path} sorted index {i}") for i in ids_to_show],
        name="attribution_search",
    )


def collect_mlp_stats(model, params, n_seqs=100):
    dataset_here = get_val_seqs()[: n_seqs // batch_size * batch_size, :seq_len]
    dataset_batches = dataset_here.reshape(-1, batch_size, seq_len)
    result = []
    for batch in tqdm(dataset_batches):
        tensor = get_mlp_stats(model, params, batch)
        tensor = np.array(tensor, dtype=np.float16)
        result.append(tensor)
    result = np.concatenate(result, axis=0)
    return result, dataset_here


def get_top_k_docs_per_neuron_activation(model, params, n_seqs=100, k=5, seq_len=100, token_aggregator="max"):
    """For each sequence/doc, take the [mean or max or min] activation across tokens to get the per-neuron score. Take the"""
    assert token_aggregator in {"mean", "max", "min"}

    val_data = get_val_seqs()[: (n_seqs // batch_size) * batch_size, :seq_len]
    val_data_batches = val_data.reshape(-1, batch_size, seq_len)
    top_k_doc_idx = np.zeros((k, model.num_layers, model.hidden_size * 4), dtype=np.int32)
    top_k_vals = -np.infty * np.ones((k, model.num_layers, model.hidden_size * 4))
    for i, batch in tqdm(enumerate(val_data_batches)):
        activations = np.array(get_mlp_stats(model, params, batch), dtype=np.float16)
        if token_aggregator == "mean":
            batch_vals = activations.mean(axis=-2)
        elif token_aggregator == "max":
            batch_vals = activations.max(axis=-2)
        elif token_aggregator == "min":
            batch_vals = activations.min(axis=-2)
        else:
            assert False, "invalid token_aggregator arg"

        # Get the new top k vals from concatenating the old top_k_vals and batch_vals
        batch_and_top_k_vals = np.concatenate((batch_vals, top_k_vals), axis=0)
        if token_aggregator == "min":
            bk_top_idx = batch_and_top_k_vals.argsort(axis=0)[:k]
        elif token_aggregator in {"max", "mean"}:
            bk_top_idx = batch_and_top_k_vals.argsort(axis=0)[-k:]
        else:
            assert False, "invalid token_aggregator arg"
        top_k_vals = np.take_along_axis(batch_and_top_k_vals, bk_top_idx, axis=0)

        new_k_idx, l_idx, n_idx = np.where(bk_top_idx >= batch_size)

        # update top_k_doc_idx in the case bk_top_idx is pointing to the old top_k_doc_idx
        top_k_doc_idx[new_k_idx, l_idx, n_idx] = top_k_doc_idx[
            bk_top_idx[new_k_idx, l_idx, n_idx] - batch_size, l_idx, n_idx
        ]
        # update top_k_doc_idx in the case bk_top_idx is pointing to indices from the batch
        top_k_doc_idx[bk_top_idx < batch_size] = (
            bk_top_idx[bk_top_idx < batch_size] + i * batch_size
        )  # add in the number of documents seen before the batch

    # Adjust doc_idx and vals to be 'oriented' correctly'
    # Move indexing over k to be the last axis
    top_k_doc_idx, top_k_vals = np.moveaxis(top_k_doc_idx, 0, -1), np.moveaxis(top_k_vals, 0, -1)
    # Make it high->low instead of low->high
    if token_aggregator in {"max", "mean"}:
        top_k_doc_idx, top_k_vals = np.flip(top_k_doc_idx, axis=-1), np.flip(top_k_vals, axis=-1)

    return top_k_doc_idx, top_k_vals, val_data


def collect_mlp_attribs(model, params, n_seqs=100):
    dataset_here = get_val_seqs()[: n_seqs // batch_size * batch_size, :seq_len]
    dataset_batches = dataset_here.reshape(-1, batch_size, seq_len)
    result = []
    for batch in tqdm(dataset_batches):
        layries = []
        for l in range(model.num_layers):
            tensor = get_mlp_attribs_unjit(model, params, l, batch)
            tensor = np.array(tensor, dtype=np.float16)
            layries.append(tensor)
        result.append(np.stack(layries, axis=1))
    result = np.concatenate(result, axis=0)
    return result, dataset_here
