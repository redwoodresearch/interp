from typing import List, Optional
from interp.model.gpt_model import Gpt
from interp.tools.grad_modify_query import ModifierCollectionTreeNode, Query, TargetConf, run_query
from functools import partial
from interp.tools.grad_modify_query_items import AddConf, ItemConf, MulConf
from interp.tools.interpretability_tools import single_tokenize, toks_to_string_list
from interp.tools.log import KeyIdxs
import jax.numpy as jnp
import jax
from flax.core.scope import FrozenVariableDict
import numpy as np


def unembed_unjit(
    model: Gpt, params: FrozenVariableDict, vector: jnp.ndarray, transpose_embed_instead_of_unembed: bool = False
):
    unembed_matrix = (
        model.bind(params).embedding.token_embedding.embedding
        if transpose_embed_instead_of_unembed
        else model.bind(params).embedding.token_unembedding.embedding
    )
    logits = jnp.einsum("th,h->t", unembed_matrix, vector)
    logprobs = jax.nn.log_softmax(logits)
    return {"logits": logits, "logprobs": logprobs}


unembed = jax.jit(unembed_unjit, static_argnames=["model", "transpose_embed_instead_of_unembed"])


def unembed_prob_direction_unjit(model: Gpt, params: FrozenVariableDict, vector: jnp.ndarray, location: jnp.ndarray):
    def loss(log):
        inp, cache = log.log_and_modify(location, "inp", cache)
        out, cache = model.bind(params).out(inp, log)
        softmaxed = jax.nn.softmax(out)
        _, cache = log.log_and_modify(softmaxed, "probs", cache)
        return log

    log = run_query(
        loss,
        Query(
            targets=[TargetConf(KeyIdxs("loss"), display_name="out")],
            modifier_collection_tree=ModifierCollectionTreeNode(AddConf(ItemConf(KeyIdxs(f"inp")), multiplier=vector)),
        ),
    )
    result = log["out"]
    return result


unembed_prob_direction = jax.jit(
    unembed_prob_direction_unjit,
    static_argnames=[
        "model",
    ],
)


def top_vocab(logits: jnp.ndarray, k: int, specific_words: List[str]):
    specific_tokens = [single_tokenize(t) for t in specific_words]
    sorted_idxs = jnp.argsort(logits)
    topk = sorted_idxs[-k:]
    topk = topk[::-1]
    bottomk = sorted_idxs[:k]
    return {
        "top": {"values": logits[topk].tolist(), "words": toks_to_string_list(topk)},
        "bottom": {"values": logits[bottomk].tolist(), "words": toks_to_string_list(bottomk)},
        "specific": {"values": logits[np.array(specific_tokens, dtype=np.int32)].tolist(), "words": specific_words},
    }


def unembed_to_topk(
    model: Gpt,
    params: FrozenVariableDict,
    tokenizer,
    vector: jnp.ndarray,
    k=20,
    specific_words: List[str] = [],
    location: Optional[jnp.ndarray] = None,
):
    vector_coerced = np.array(vector)
    obj = unembed(model, params, vector_coerced)
    if location is not None:
        obj["probs"] = unembed_prob_direction(model, params, vector_coerced, location)
    tops = {logit_type: top_vocab(v, k, specific_words) for logit_type, v in obj.items()}
    return tops["logprobs"]
