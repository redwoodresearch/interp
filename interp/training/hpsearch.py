import numpy as np
from qa.search import Search

mysearch = Search(
    function_path="interp.training.training.train_and_evaluate_autoregressive",
    grid={
        "train_and_evaluate_autoregressive.model_config": [
            dict(
                hidden_size=256,
                num_heads=8,
                num_layers=2,
                vocab_size=50259,
                norm_type="layer_norm",
                pos_enc_type="gpt",
                use_mlp=True,
                use_norm_output=True,
                max_sequence_len=2048,
                tied_embed_unembed=True,
            )
        ],
        "train_and_evaluate_autoregressive.lr": np.geomspace(4e-5, 1e-3, 3),
        "train_and_evaluate_autoregressive.batch_tokens": [12000, 24000],
        "train_and_evaluate_autoregressive.n_files": [4],
        "train_and_evaluate_autoregressive.model_name": ["small_transformers"],
    },
    random={},
    gin_config="blank.gin",
)
