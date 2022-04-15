from functools import lru_cache
from typing import Dict, List, Any, Tuple
import os
import json

import transformers
import jax
from flax import serialization
from flax.core.scope import FrozenVariableDict
import jax.numpy as jnp

from interp.model.gpt_model import Gpt, gpt_init
from interp.tools.interpretability_tools import get_interp_tokenizer
from interp.tools.rrfs import RRFS_DIR
import interp.tools.optional as op
from functools import partial

RRFS_INTERP_MODELS_DIR = f"{RRFS_DIR}/interpretability_models_jax/"
MODELS_DIR = os.environ.get("INTERPRETABILITY_MODELS_DIR", RRFS_INTERP_MODELS_DIR)
MODEL_STATS_PATH = (
    os.path.expanduser("~/interpretability_model_stats")
    if os.path.exists(os.path.expanduser("~/interpretability_model_stats"))
    else f"{RRFS_DIR}/interpretability_model_stats"
)


def get_gpt_tokenizer_with_end_tok():
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokenizer._add_tokens(["[END]"])
    tokenizer.pad_token = "[END]"
    return tokenizer


MODEL_CLASS_STR_TO_MODEL_AND_TOKENIZER_FNS: Dict[str, List[Any]] = {
    "GPTBeginEndToks": [Gpt, get_interp_tokenizer],
    "GPT": [Gpt, get_gpt_tokenizer_with_end_tok],
}

MODEL_IDS = list(MODEL_CLASS_STR_TO_MODEL_AND_TOKENIZER_FNS.keys())


# disabling loading the params gets you a random model with the same config
def load_model(
    model_id,
    models_dir=MODELS_DIR,
    load_params=True,
    dtype=None,
    key: jax.random.KeyArray = None,
) -> Tuple[Any, FrozenVariableDict, Any]:
    print(f"Loading models from {models_dir}")

    key = op.unwrap_or(key, jax.random.PRNGKey(0))

    p = models_dir + "/" + model_id
    if not os.path.exists(p):
        print(
            f"Can't find model at {p}, falling back to {MODELS_DIR}/{model_id}. Cache it locally with `rsync -r {MODELS_DIR} {models_dir}"
        )
        p = MODELS_DIR + "/" + model_id
    print(f"loading {model_id} from {p}")
    model_info = json.load(open(p + "/model_info.json", "r"))
    model_class, get_tokenizer = MODEL_CLASS_STR_TO_MODEL_AND_TOKENIZER_FNS[model_info["model_class"]]
    model_class_args = model_info["model_config"]
    if dtype is not None:
        model_class_args["dtype"] = dtype
    model = model_class(**model_class_args)
    init_params = gpt_init(model, key)
    if load_params:
        params = serialization.from_bytes(init_params, open(p + "/model.bin", "rb").read())
    else:
        params = init_params
    if dtype is not None:
        params = jax.tree_util.tree_map(partial(jnp.array, dtype=dtype), params)
    else:
        params = jax.tree_util.tree_map(jnp.array, params)
    tokenizer = get_tokenizer()

    return model, params, tokenizer


def save_model(model: Gpt, params, name, model_class, desc, extra_info={}, models_dir=MODELS_DIR):
    path = f"{models_dir}/{name}"
    if not os.path.exists(path):
        os.mkdir(path)
    model_info = {
        "model_config": {
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "vocab_size": model.vocab_size,
            "hidden_size": model.hidden_size,
            "max_sequence_len": model.max_sequence_len,
            "dropout_rate": model.dropout_rate,
            "embed_dropout_rate": model.embed_dropout_rate,
            "attn_probs_dropout_rate": model.attn_probs_dropout_rate,
            "norm_type": model.norm_type,
            "attn_bias": model.attn_bias,
            "layer_norm_epsilon": model.layer_norm_epsilon,
            "pos_enc_type": model.pos_enc_type,
            "use_mlp": model.use_mlp,
            "use_norm_output": model.use_norm_output,
        },
        "model_class": model_class,
        "attn_path_info": ["blocks", 12, "attention"],
        "mlp_path_info": ["blocks", 12, ""],
        "attn_tensor_names": ["attn_prob", "weighted_attn_prob"],
        "mlp_tensor_names": ["post_act", "pre_act"],
        "desc": desc,
        "extra_info": extra_info,
    }
    json.dump(model_info, open(path + "/model_info.json", "w"))
    open(path + "/model.bin", "wb").write(serialization.to_bytes(params))
    print("saved model")


def get_model_callable(model, params):
    return jax.jit(lambda toks: model.apply(params, toks)[0])


@lru_cache()
def get_dataset_mean_activations(model_name):
    """
    returns {
        "mlps": jnp.ndarray[num_layers, num_neurons, hidden_size],
        "heads": jnp.ndarray[num_layers, num_heads, hidden_size],
        # "layer_inputs":jnp.ndarray[num_layers,hidden_size]
    }
    """
    if os.path.exists(f"{MODEL_STATS_PATH}/{model_name}/mean_activations"):
        return serialization.msgpack_restore(open(f"{MODEL_STATS_PATH}/{model_name}/mean_activations", "rb").read())
    return None
