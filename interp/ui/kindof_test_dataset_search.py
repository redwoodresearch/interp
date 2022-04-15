import jax
from interp.model.model_loading import load_model
from interp.tools.grad_modify_query import ModifierCollectionTreeNode, Query, TargetConf, run_query
from interp.tools.grad_modify_query_items import ItemConf, MulConf
from interp.tools.grad_modify_query_utils import compose_trees
from interp.tools.log import Logger
from interp.tools.data_loading import get_val_seqs
import numpy as np
import jax.numpy as jnp
import interp.ui.dataset_search as ds


model, params, tokenizer = load_model("jan5_attn_only_two_layers")


def test_thing():
    x = ds.get_max_path_contribution(model, params, ((1, 5, "O"),))


if __name__ == "__main__":
    test_thing()
