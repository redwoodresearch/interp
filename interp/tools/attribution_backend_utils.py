from typing import List, Literal, Optional, Union

import jax.numpy as jnp
from interp.model.gpt_model import Gpt
from interp.model.grad_modify_fake import ablation_softmax_probs, half_linear_gelu, integrated_gradients_softmax_probs
from interp.tools.assert_never import assert_never

from interp.tools.custom_jvp import ablation_custom_jvp, integrated_gradients_custom_jvp
from interp.tools.grad_modify_query import ModifierCollectionTreeNode
from interp.tools.grad_modify_query_items import StopGradConf, ItemConf
from interp.tools.log import EnableModSetup, Idxs, KeyIdxs


def stop_qkv_except(
    layer_idxs: Idxs,
    qkv_except: Union[int, Literal["q", "k", "v"], jnp.ndarray],
    static=True,
):
    if isinstance(qkv_except, str):
        qkv_except = "qkv".index(qkv_except)

    if isinstance(qkv_except, int):
        assert 0 <= qkv_except < 3
        enable_by_idx = False
    else:
        assert qkv_except.ndim in {0, 1}
        enable_by_idx = qkv_except.ndim == 1

    return tuple(
        ModifierCollectionTreeNode(
            StopGradConf(
                ItemConf(
                    KeyIdxs(f"blocks.attention.{l}", layer_idxs),
                    enable_setup=EnableModSetup(i != qkv_except, is_enable_static=static, enable_by_idx=enable_by_idx),
                ),
            )
        )
        for i, l in enumerate("qkv")
    )


# ig = integrated_gradients
Fake = Literal["none", "ablation", "ig"]

FAKE_VALUES: List[Fake] = ["none", "ablation", "ig"]


def fake_to_wrapper(fake: Fake):
    if fake == "ig":
        return integrated_gradients_custom_jvp
    elif fake == "ablation":
        return ablation_custom_jvp
    elif fake == "none":
        return lambda x: x
    else:
        assert_never(fake)


def fake_to_attn_probs(fake_attn_probs: Fake, idxs: Idxs = Idxs.all()) -> Optional[ModifierCollectionTreeNode]:
    if fake_attn_probs == "ig":
        return integrated_gradients_softmax_probs(idxs)
    elif fake_attn_probs == "ablation":
        return ablation_softmax_probs(idxs)
    elif fake_attn_probs == "none":
        return None
    else:
        assert_never(fake_attn_probs)


def fake_to_mlp_activation(fake_attn_probs: Fake, idxs: Idxs = Idxs.all()) -> Optional[ModifierCollectionTreeNode]:
    if fake_attn_probs == "half_linear":
        return half_linear_gelu(fraction=0.3, layer_idxs=idxs)
    elif fake_attn_probs == "none":
        return None
    elif fake_attn_probs == "ablation" or fake_attn_probs == "ig":
        raise Exception("mlp doesn't support ablation or ig yet")
    else:
        assert_never(fake_attn_probs)
