# %%

import os

import jax
import jax.numpy as jnp

from interp.model.gpt_model import Gpt
from interp.model.model_loading import load_model
from interp.model.monte_carlo import monte_carlo_generative
from interp.tools.interpretability_tools import get_interp_tokenizer, sequence_tokenize
import interp.tools.optional as op

# %%

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# %%

models_dir_local = os.path.expanduser("~/interp_models_jax")
gpt2, gpt2_vars, _ = load_model("gpt2", models_dir=models_dir_local)
gpt2: Gpt = gpt2
gpt2_b = gpt2.bind(gpt2_vars)

# %%

print(gpt2_b.get_qkv_mats_all_layers().shape)
print(gpt2_b.get_q_mats_all_layers().shape)
print(gpt2_b.get_k_mats_all_layers().shape)
print(gpt2_b.get_v_mats_all_layers().shape)
print(gpt2_b.get_qk_mats_all_layers().shape)
print(gpt2_b.get_o_mats_all_layers().shape)
print(gpt2_b.get_qk_combined_mats_all_layers().shape)
print(gpt2_b.get_ov_combined_mats_all_layers().shape)

# %%

toks = gpt2_vars["params"]["embedding"]["token_embedding"]["embedding"][:100]
pos = gpt2_vars["params"]["embedding"]["position_embedding"]["embedding"][:30]
print(toks.shape)
print(pos.shape)
print(gpt2_b.compute_attn_scores_all_layers(toks, pos, mask=False).shape)

# %%

print(
    get_interp_tokenizer().decode(
        op.unwrap(
            monte_carlo_generative(
                jax.random.PRNGKey(0),
                gpt2,
                gpt2_vars,
                n_toks=20,
                n_samples=1,
                batch_size=1,
                prompt=sequence_tokenize("[BEGIN] Hi Sally, I'm about to"),
                pad=True,
            ).toks
        )[0]
    )
)
