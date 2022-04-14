from typing import Tuple

import pytest
from flax.core.scope import FrozenVariableDict
import jax
import jax.numpy as jnp

from interp.model.gpt_model import Gpt, gpt_init
from interp.model.model_loading import load_model
from interp.tools.interpretability_tools import batch_tokenize, begin_token

ModelParams = Tuple[Gpt, FrozenVariableDict]

# everything here should be constant, so "session" scope is for efficiency


@pytest.fixture(scope="session")
def loaded_model() -> ModelParams:
    model, params, _ = load_model("jan5_attn_only_two_layers/")

    return model, params


@pytest.fixture(scope="session")
def loaded_model_random_params(loaded_model: ModelParams) -> FrozenVariableDict:
    return random_model_setup(loaded_model[0])[-1]


def random_model_setup(model: Gpt, key=jax.random.PRNGKey(5)) -> ModelParams:
    return model, gpt_init(model, key)


@pytest.fixture(scope="session")
def tiny_random_model() -> ModelParams:
    return random_model_setup(Gpt(num_heads=4, hidden_size=64, vocab_size=5000))


@pytest.fixture(scope="session")
def tiny_tiny_random_model() -> ModelParams:
    # now this is what transforming is all about
    return random_model_setup(Gpt(num_heads=2, hidden_size=4, pos_enc_type="shortformer", vocab_size=30))


@pytest.fixture(scope="session")
def random_model_tiny_mlps() -> ModelParams:
    return random_model_setup(Gpt(use_mlp=True, num_heads=4, hidden_size=64, vocab_size=5000))


@pytest.fixture(scope="session")
def random_model_tiny_relu_layer_norm() -> ModelParams:
    return random_model_setup(
        Gpt(
            use_mlp=True,
            mlp_act_type="relu",
            norm_type="layer_norm",
            use_norm_output=True,
            attn_bias=True,
            num_heads=2,
            hidden_size=48,
            vocab_size=5000,
        )
    )


@pytest.fixture(scope="session")
def example_seq_no_begin():
    text = " \"I don't sleep right,\" Harry said. He waved his hands helplessly. \"My sleep cycle is twenty-six hours long, I always go to sleep two hours later, every day. I can't fall asleep any earlier than that, and then the next day I go to sleep two hours later than that. 10PM, 12AM, 2AM, 4AM, until it goes around the clock. Even if I try to wake up early, it makes no difference and I'm a wreck that whole day. That's why I haven't been going to a normal school up until now.\""
    return batch_tokenize([text])


@pytest.fixture(scope="session")
def example_seq(example_seq_no_begin: jnp.ndarray):
    return jnp.concatenate([jnp.expand_dims(begin_token(), (0, 1)), example_seq_no_begin], axis=1)
