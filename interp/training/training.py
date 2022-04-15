from comet_ml import Experiment

from interp.model.gpt_model import Gpt, gpt_init
from interp.model.model_loading import save_model
from interp.tools.interpretability_tools import get_interp_tokenizer
from interp.tools.data_loading import get_train_full_length_seqs, get_val_seqs, to_batches, to_binned_batches
import optax
import jax
import jax.numpy as jnp
from tqdm import trange
import gin


def train_autoregressive_gpt(model: Gpt, params, lr, n_files=None, tokens_per_batch=24000, experiment=None):
    train_seqs = get_train_full_length_seqs(n_files=n_files)
    batches = to_binned_batches(train_seqs, tokens_per_batch=tokens_per_batch, max_length=2047)
    n_steps = len(batches)
    warmup_steps = 100

    def schedule_fn(i):
        if i < warmup_steps:
            return i / warmup_steps
        frac = (i - warmup_steps) / (n_steps - warmup_steps)
        scale = 1 - frac
        experiment.log_metric("lr", scale * lr)
        return 1 - frac

    optimizer = optax.chain(optax.scale_by_schedule(schedule_fn), optax.adam(learning_rate=lr))
    opt_state = optimizer.init(params)

    def loss_raw_fn(params, toks, rng_key):
        logits = model.apply(params, toks, rngs={"dropout": rng_key}, training=True)
        logprobs = jax.nn.log_softmax(logits, axis=-1)
        logprobs_predicting_next = jnp.take_along_axis(
            logprobs,
            jnp.expand_dims(jnp.concatenate([toks[:, 1:], jnp.full((toks.shape[0], 1), 0)], 1), axis=-1),
            2,
        )[:, :, 0]
        loss = -jnp.mean(logprobs_predicting_next)
        return loss

    random_key = jax.random.PRNGKey(0)
    loss_grad_fn = jax.jit(jax.value_and_grad(loss_raw_fn))

    def params_and_tokens_to_params(params, opt_state, batch, key):
        loss_val, grads = loss_grad_fn(params, batch, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss_val, params, opt_state

    pbar = trange(len(batches))
    for i in pbar:
        subkey, random_key = jax.random.split(random_key)
        batch = batches[i]
        loss_val, params, opt_state = params_and_tokens_to_params(params, opt_state, batch, subkey)
        pbar.set_description(f"Loss: {str(loss_val.tolist())[:6]}")
        if experiment is not None:
            experiment.log_metric("loss", loss_val, step=i)
    return params


def evaluate_autoregressive_gpt(model: Gpt, params, n_files=10, batch_size=128, experiment=None):
    train_seqs = get_val_seqs(train=False, n_files=n_files)
    batched_train_seqs = to_batches(train_seqs, batch_size)
    print("num training tokens", batched_train_seqs.size)

    @jax.jit
    def loss_raw_fn(params, toks, rng_key):
        logits = model.apply(params, toks, rngs={"dropout": rng_key}, training=False)
        logprobs = jax.nn.log_softmax(logits, axis=-1)
        logprobs_predicting_next = jnp.take_along_axis(
            logprobs,
            jnp.expand_dims(jnp.concatenate([toks[:, 1:], jnp.full((toks.shape[0], 1), 0)], 1), axis=-1),
            2,
        )[:, :, 0]
        loss = -jnp.mean(logprobs_predicting_next)
        return loss

    random_key = jax.random.PRNGKey(0)
    pbar = trange(batched_train_seqs.shape[0])
    losses = []
    for i in pbar:
        subkey, random_key = jax.random.split(random_key)
        batch = batched_train_seqs[i]
        loss_val = loss_raw_fn(params, batch, subkey)
        pbar.set_description(f"Loss: {str(loss_val.tolist())[:6]}")
        losses.append(loss_val)
    loss = jnp.mean(jnp.stack(losses))
    print(f"Validation loss: {loss}")
    if experiment is not None:
        experiment.log_metric("val_loss", loss)
    return params


@gin.configurable
def train_and_evaluate_autoregressive(
    model_name, model_config, lr, batch_tokens, description="blank description", n_files=None
):
    experiment = Experiment(
        api_key="vABV7zo6pqS7lfzZBhyabU2Xe",
        project_name=model_name,
        workspace="redwood",
    )
    experiment.log_parameter("model_config", repr(model_config))
    experiment.log_parameter("training_config", repr(model_config))
    experiment.log_parameter("lr", lr)
    model = Gpt(**model_config)
    params = gpt_init(model, jax.random.PRNGKey(0))
    params = train_autoregressive_gpt(
        model, params, experiment=experiment, n_files=n_files, lr=lr, tokens_per_batch=batch_tokens
    )
    save_model(model, params, model_name + experiment.get_key(), "GPTBeginEndToks", description)
    print("evaluating")
    evaluate_autoregressive_gpt(model, params, experiment=experiment)


if __name__ == "__main__":
    max_seq_len = 2048
    tokenizer_len = len(get_interp_tokenizer())
    train_and_evaluate_autoregressive(
        "apr6_2l_attn_only_untied",
        dict(
            hidden_size=256,
            num_heads=8,
            num_layers=2,
            vocab_size=20259,
            norm_type="layer_norm",
            pos_enc_type="shortformer",
            use_mlp=False,
            use_norm_output=True,
            max_sequence_len=max_seq_len,
            tied_embed_unembed=False,
            mlp_act_type="relu",
            attn_bias=True,
        ),
        lr=2.5e-4,
        n_files=None,
        batch_tokens=30000,
    )
