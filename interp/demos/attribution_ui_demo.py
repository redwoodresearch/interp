# %%
import os

from attrs import asdict

from interp.model.gpt_model import Gpt
from interp.tools.interpretability_tools import batch_tokenize, single_tokenize, toks_to_string_list
from interp.model.model_loading import load_model
from interp.ui.attribution_backend import AttributionBackend
from interp.ui.very_named_tensor import VeryNamedTensor

# %%

models_dir_local = os.path.expanduser("~/interp_models_jax")
model, params, tokenizer = load_model("jan11_gpt_2l/", models_dir=models_dir_local)
# model, params, tokenizer = load_model("gpt2_jax/", models_dir=models_dir_local)
model: Gpt = model

# %%


s = "[BEGIN] Welcome to the Redwood Interpretability site. The Redwood Interpretability team"
backend = AttributionBackend(model, params, tokenizer, s)
new_seq = batch_tokenize([s])

# %%

seq_idx = 101
print(toks_to_string_list(new_seq[0])[seq_idx])
target_tok_str = " Interpret"

# %%

(new_seq == single_tokenize(target_tok_str)).nonzero()

# %%

vnt_start = backend.startTree(
    {"kind": "logprob", "data": {"seqIdx": seq_idx, "tokString": target_tok_str, "comparisonTokString": None}},
    False,
    False,
)
vnt_start["mlps"][0, 1, :, seq_idx]

# %%


# %%

vnt_expand = backend.expandTreeNode([(2, 4, seq_idx)], False)
assert isinstance(vnt_expand, VeryNamedTensor)
indexed = vnt_expand["layer_0", 0, :, "k"].tensor
indexed.max(), indexed.argmax()

# %%

vnt_expand["layer_0", 0, :, "k"]

# %%

layer = 0
vnt_expand_after = backend.expandTreeNode([(2, 4, seq_idx, 1), (1, 0, indexed.argmax())], False)
assert isinstance(vnt_expand_after, VeryNamedTensor)
vnt_expand_after[0, 0, :, "v"]

# %% [markdown]
# Indirect attributions:

# %%
vnt_start = backend.startTree(
    {
        "kind": "logprob",
        "data": {"seqIdx": seq_idx, "tokString": target_tok_str, "comparisonTokString": None},
        "direct": "False",
    },
    False,
    False,
)
assert isinstance(vnt_start, VeryNamedTensor)
vnt_start[:, :, seq_idx]
# %%
vnt_expand = backend.expandTreeNode([{"layerWithIO": 2, "token": seq_idx, "isMlp": False, "headOrNeuron": 4}], False)
assert isinstance(vnt_expand, VeryNamedTensor)
indexed = vnt_expand["layer_0", 0, :, "k"].tensor
indexed.max(), indexed.argmax()
# %%
layer = 0
vnt_expand_after = backend.expandTreeNode(
    [
        {"layerWithIO": 2, "token": seq_idx, "isMlp": False, "headOrNeuron": 4},
        {"layerWithIO": 1, "token": seq_idx, "isMlp": False, "headOrNeuron": 0},
    ],
    False,
)
assert isinstance(vnt_expand_after, VeryNamedTensor)
vnt_expand_after[0, 0, :, "v"]
