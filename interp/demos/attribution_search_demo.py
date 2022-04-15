# %%
import os

from interp.ui.attribution_backend import AttributionBackend
from interp.ui.attribution_backend_comp import LayerType
from interp.ui.very_named_tensor import VeryNamedTensor
from interp.model.gpt_model import Gpt
from interp.model.model_loading import load_model
from interp.tools.interpretability_tools import batch_tokenize, single_tokenize, toks_to_string_list

# %%

models_dir_local = os.path.expanduser("~/interp_models_jax")
model, params, tokenizer = load_model("gpt2/", models_dir=models_dir_local)
# model, params, tokenizer = load_model("gpt2_jax/", models_dir=models_dir_local)
model: Gpt = model

# %%
s = '[BEGIN] "I love Mrs. Dursley. I don\'t sleep right," Harry said. He waved his hands helplessly. "Mrs. Dursley\'s sleep cycle is twenty-six hours long, I always go to sleep two hours later, every day.'
# s = "[BEGIN] The office of the Deputy Headmistress was clean and well-organised; on the wall immediately adjacent to the desk was a maze of wooden cubbyholes of all shapes and sizes, most with several parchment scrolls thrust into them, and it was somehow very clear that Professor McGonagall knew exactly what every cubbyhole meant, even if no one else did. A single parchment lay on the actual desk, which was, aside from that, clean. Behind the desk was a closed door barred with several locks. In Hogwarts"
backend = AttributionBackend(model, params, tokenizer, s)
new_seq = batch_tokenize([s])

# %%

seq_idx = 30
print(toks_to_string_list(new_seq[0])[seq_idx])
target_tok_str = "ley"

# %%

(new_seq == single_tokenize(target_tok_str)).nonzero()
# %%

vnt_start = backend.startTree(
    {"kind": "logprob", "data": {"seqIdx": seq_idx, "tokString": target_tok_str, "comparisonTokString": None}},
    False,
    False,
)
assert isinstance(vnt_start["heads"], VeryNamedTensor)
vnt_start["heads"][0, :, :, seq_idx]

# %%
nodes, node_attribs, _ = backend.searchAttributionsFromStart(0.00001)
# %%
edges = backend.getAttributionInMask(10, use_neg=True)
edges
# %%
