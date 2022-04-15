from typing import Tuple
from dataclasses import dataclass
from typing import Any, Callable, List, Optional
import time
from interp.tools.interpretability_tools import get_interp_tokenizer
import interp.tools.optional as op
import jax
import jax.numpy as jnp
import numpy as np


def index_to_string(index):
    fixed_list = []
    for x in index:
        if isinstance(x, slice):
            if x.start is None and x.stop is None:
                fixed_list.append(":")
            else:
                fixed_list.append(
                    ("" if x.start is None else str(x.start)) + ":" + ("" if x.stop is None else str(x.stop))
                )
        elif isinstance(x, str):
            fixed_list.append(f'"{x}"')
        else:
            fixed_list.append(str(x))
    return f"[{', '.join(fixed_list)}]"


@dataclass
class VeryNamedTensor:
    reduction_fns = {"mean": jnp.nanmean, "sum": jnp.nansum, "norm": jnp.linalg.norm, "max": jnp.amax, "min": jnp.amin}

    def __init__(self, tensor, dim_names, dim_types, dim_idx_names, units, title="untitled_tensor"):
        self.tensor: jnp.ndarray = tensor
        self.dim_names: List[str] = dim_names
        self.dim_types: List[str] = dim_types
        self.dim_idx_names: List[List[str]] = dim_idx_names
        self.units: str = units
        self.title = title
        self.shape = self.tensor.shape
        assert len(dim_names) == len(
            self.shape
        ), f"{len(dim_names)} dim_names given ({dim_names}), but tensor is of rank {len(self.shape)} ({self.shape})"
        assert len(dim_types) == len(
            self.shape
        ), f"{len(dim_types)} dim_types given ({dim_types}), but tensor is of rank {len(self.shape)} ({self.shape})"
        idx_shape = tuple([len(g) for g in self.dim_idx_names])
        assert (
            idx_shape == self.shape
        ), f"dim_idx_names of shape {idx_shape} does not match tensor of shape {self.shape}"

    # can index with slices of numbers, or names
    # doesn't support
    def __getitem__(self, axes):
        return self.getitem(axes, False)

    def getitem(self, axes, no_title_change=False):

        if not isinstance(axes, tuple):
            axes = (axes,) + (slice(None, None),) * (len(self.dim_types) - 1)

        if len(axes) < len(self.dim_types):
            axes = axes + (slice(None, None),) * (len(self.dim_types) - len(axes))

        axes = tuple([self.dim_idx_names[i].index(x) if isinstance(x, str) else x for i, x in enumerate(axes)])

        return VeryNamedTensor(
            tensor=self.tensor.__getitem__(axes),
            dim_names=[x for i, x in zip(axes, self.dim_names) if isinstance(i, slice)],
            dim_types=[x for i, x in zip(axes, self.dim_types) if isinstance(i, slice)],
            dim_idx_names=[x.__getitem__(i) for i, x in zip(axes, self.dim_idx_names) if isinstance(i, slice)],
            units=self.units,
            title=self.title if no_title_change else f"{self.title} at {index_to_string(axes)}",
        )

    def getView(self, view, truncate_nan_rows=False):
        """
        view is list, where every element is either:
            - 'axis' or 'facet', representing no change
            - int, representing index for that dim
            - str in self.reduction_fns.keys(), representing reduction fn
        """
        result_tensor = self.tensor
        for i, view_el in enumerate(view):
            if view_el in ["axis", "facet"]:
                continue
            elif isinstance(view_el, str):
                assert (
                    view_el in self.reduction_fns.keys()
                ), f"element {view_el} in view {view} not supported, options are {list(self.reduction_fns.keys())}"
                reduction_fn: Callable = op.unwrap(self.reduction_fns.get(view_el))
                result_tensor = reduction_fn(result_tensor, axis=i, keepdims=True)
            elif isinstance(view_el, int):
                result_tensor = jnp.take(result_tensor, jnp.array([view_el]), i)
            else:
                raise ValueError(f"Invalid view: {view}")
        for i, view_el in enumerate(reversed(view)):
            if view_el not in ["axis", "facet"]:
                result_tensor = result_tensor.squeeze(axis=len(view) - i - 1)
        result_vnt = self.__getitem__(tuple([slice(None, None) if x in ["axis", "facet"] else 0 for x in view]))
        result_vnt.tensor = result_tensor
        return result_vnt.truncate_nan_rows() if truncate_nan_rows else result_vnt

    def truncate_nan_rows(self):
        is_nan = jnp.isnan(self.tensor)
        all_nan_by_dim_row = [
            jnp.array([jnp.take(is_nan, jnp.array([i]), dim).all() for i in range(self.shape[dim])])
            for dim in range(len(self.shape))
        ]
        any_row_all_nan = np.any([all_nan_by_row.any() for all_nan_by_row in all_nan_by_dim_row])
        if any_row_all_nan:
            new_tensor = self.tensor
            new_dim_idx_names = self.dim_idx_names
            for dim, rows_all_nan in enumerate(all_nan_by_dim_row):
                if rows_all_nan.any():
                    not_nan_idxs = jnp.arange(len(rows_all_nan))[~rows_all_nan]
                    new_tensor = jnp.take(new_tensor, not_nan_idxs, dim)
                    new_dim_idx_names[dim] = [self.dim_idx_names[dim][i] for i in not_nan_idxs]
            return VeryNamedTensor(
                tensor=new_tensor,
                dim_names=self.dim_names,
                dim_types=self.dim_types,
                dim_idx_names=new_dim_idx_names,
                units=self.units,
            )
        else:
            return self

    def to_dict(self):
        return {
            "tensor": self.tensor,
            "dim_names": self.dim_names,
            "dim_idx_names": self.dim_idx_names,
            "dim_types": self.dim_types,
            "units": self.units,
            "title": self.title,
        }

    def to_lvnt(self):
        return LazyVeryNamedTensor(
            lambda: self.tensor,
            dim_names=self.dim_names,
            dim_types=self.dim_types,
            dim_idx_names=self.dim_idx_names,
            units=self.units,
            title=self.title,
        )

    def __repr__(self):
        indent = " " * 4  # len("VeryNamedTensor(")
        str = f"VeryNamedTensor(\n"
        str += f"{indent}shape={self.shape}\n"
        str += f"{indent}dim_names={self.dim_names}\n"
        str += f"{indent}dim_types={self.dim_types}\n"
        str += f"{indent}dim_idx_names={self.dim_idx_names}\n"
        str += f"{indent}units={self.units}\n"
        str += f"{indent}tensor={self.tensor}\n"
        str += ")"
        return str


class LazyVeryNamedTensor:
    def __init__(self, tensor_thunk, dim_names, dim_types, dim_idx_names, units, title="untitled_tensor"):
        self.tensor_thunk: Callable[[], jnp.ndarray] = tensor_thunk
        self.dim_names: List[str] = dim_names
        self.dim_types: List[str] = dim_types
        self.dim_idx_names: List[List[str]] = dim_idx_names
        self.units: str = units
        self.title = title
        self.shape = tuple([len(x) for x in dim_idx_names])

        self.realized_vnt: Optional[VeryNamedTensor] = None

        assert len(dim_names) == len(
            self.shape
        ), f"{len(dim_names)} dim_names given ({dim_names}), but tensor is of rank {len(self.shape)} ({self.shape})"
        assert len(dim_types) == len(
            self.shape
        ), f"{len(dim_types)} dim_types given ({dim_types}), but tensor is of rank {len(self.shape)} ({self.shape})"
        idx_shape = tuple([len(g) for g in self.dim_idx_names])
        assert (
            idx_shape == self.shape
        ), f"dim_idx_names of shape {idx_shape} does not match tensor of shape {self.shape}"

    def __getitem__(self, axes):
        return self.getitem(axes, False)

    def getitem(self, axes: Tuple, no_title_change: bool = False):
        if not isinstance(axes, tuple):
            axes = (axes,) + (slice(None, None),) * (len(self.dim_types) - 1)

        if len(axes) < len(self.dim_types):
            axes = axes + (slice(None, None),) * (len(self.dim_types) - len(axes))

        axes = tuple([self.dim_idx_names[i].index(x) if isinstance(x, str) else x for i, x in enumerate(axes)])

        if self.realized_vnt is not None:
            thunk = lambda: self.realized_vnt.tensor.__getitem__(axes)
        else:
            thunk = lambda: self.tensor_thunk().__getitem__(axes)
        return LazyVeryNamedTensor(
            tensor_thunk=thunk,
            dim_names=[x for i, x in zip(axes, self.dim_names) if isinstance(i, slice)],
            dim_types=[x for i, x in zip(axes, self.dim_types) if isinstance(i, slice)],
            dim_idx_names=[x.__getitem__(i) for i, x in zip(axes, self.dim_idx_names) if isinstance(i, slice)],
            units=self.units,
            title=self.title if no_title_change else f"{self.title} at {index_to_string(axes)}",
        )

    def getView(self, *args, **kwargs):
        if self.realized_vnt is None:
            self.realized_vnt = VeryNamedTensor(
                tensor=self.tensor_thunk(),
                dim_names=self.dim_names,
                dim_types=self.dim_types,
                dim_idx_names=self.dim_idx_names,
                units=self.units,
                title=self.title,
            )
        return self.realized_vnt.getView(*args, **kwargs)

    def to_dict(self):
        return {
            "_getView": self.getView,
            "_getSparseView": lambda x: x,  # TODO reimplement this
            "dim_names": self.dim_names,
            "dim_idx_names": self.dim_idx_names,
            "dim_types": self.dim_types,
            "units": self.units,
            "title": self.title,
        }


def vnt_guessing_shit_model_tokens(tensor, model, tokens, title="title", permissive=True):
    return vnt_guessing_shit(
        tensor,
        [
            {"type": "seq", "name": "seq", "idx_names": [get_interp_tokenizer().decode([x]) for x in tokens]},
            {"type": "layer", "name": "layer", "idx_names": [str(i) for i in range(model.num_layers)]},
            {
                "type": "layerWithIO",
                "name": "layerWithIO",
                "idx_names": (["embeds"] + [str(i) for i in range(model.num_layers)] + ["outputs"]),
            },
            {"type": "heads", "name": "heads", "idx_names": [str(i) for i in range(model.num_heads)]},
            {"type": "neurons", "name": "neurons", "idx_names": [str(i) for i in range(model.hidden_size * 4)]},
            {"type": "hidden", "name": "hidden", "idx_names": [str(i) for i in range(model.hidden_size)]},
        ],
        title=title,
        permissive=permissive,
    )


def vnt_guessing_shit(tensor, potential_dims, title="title", units="units", permissive=True):
    dim_types = []
    dim_idx_names = []
    dim_names = []
    for dim_length in tensor.shape:
        matching_potentials = [x for x in potential_dims if len(x["idx_names"]) == dim_length]
        if len(matching_potentials) == 0:
            assert permissive == True, "no matching length dims"
            dim_idx_names.append([str(i) for i in range(dim_length)])
            dim_types.append("unknown")
            name = "unknown"
            while name in dim_names:
                name += "again"
            dim_names.append(name)
        else:
            matching = matching_potentials[0]
            dim_types.append(matching["type"])
            dim_idx_names.append(matching["idx_names"])
            name = matching["name"]
            while name in dim_names:
                name += "again"
            dim_names.append(name)
    return VeryNamedTensor(tensor, dim_names, dim_types, dim_idx_names, units, title)


def smaller_example_vnt():
    return VeryNamedTensor(
        tensor=jax.random.normal(jax.random.PRNGKey(2), (2, 9, 9, 3, 3)),
        dim_names=["layer", "head_to", "head_from", "q", "k"],
        dim_types=["layer", "head", "head", "seq", "seq"],
        dim_idx_names=[
            ["layer_0", "layer_1"],
            ["residual", "0", "1", "2", "3", "4", "5", "6", "7"],
            ["residual", "0", "1", "2", "3", "4", "5", "6", "7"],
            ["[BEGIN]", " Hi", " my"],
            ["[BEGIN]", " Hi", " my"],
        ],
        units="standard units",
    )


def get_big_example_vnt():
    return VeryNamedTensor(
        tensor=jax.random.normal(jax.random.PRNGKey(3), (14, 14, 12, 12, 33, 33, 3)),
        title="dummy_gradflow",
        units="grad",
        dim_names=["layer_to", "layer_from", "head_to", "head_from", "seq_to", "seq_from", "qkv_to"],
        dim_types=["layer", "layer", "head", "head", "seq", "seq", "qkv"],
        dim_idx_names=[
            [
                "embeds",
                "layer_0",
                "layer_1",
                "layer_2",
                "layer_3",
                "layer_4",
                "layer_5",
                "layer_6",
                "layer_7",
                "layer_8",
                "layer_9",
                "layer_10",
                "layer_11",
                "output",
            ],
            [
                "embeds",
                "layer_0",
                "layer_1",
                "layer_2",
                "layer_3",
                "layer_4",
                "layer_5",
                "layer_6",
                "layer_7",
                "layer_8",
                "layer_9",
                "layer_10",
                "layer_11",
                "output",
            ],
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
            [
                "[BEGIN]",
                ",",
                " Hi",
                ",",
                " my",
                " favorite",
                " dog",
                " breed",
                " is",
                " Great",
                " Dane",
                ",",
                " although",
                " I",
                " also",
                " love",
                " g",
                "erman",
                " she",
                "pher",
                "ds",
                " and",
                " pretty",
                " much",
                " any",
                " other",
                " dog",
                " breed",
                " you",
                " could",
                " possibly",
                " imagine",
                "!",
            ],
            [
                "[BEGIN]",
                ",",
                " Hi",
                ",",
                " my",
                " favorite",
                " dog",
                " breed",
                " is",
                " Great",
                " Dane",
                ",",
                " although",
                " I",
                " also",
                " love",
                " g",
                "erman",
                " she",
                "pher",
                "ds",
                " and",
                " pretty",
                " much",
                " any",
                " other",
                " dog",
                " breed",
                " you",
                " could",
                " possibly",
                " imagine",
                "!",
            ],
            ["q", "k", "v"],
        ],
    )


def get_topk_sparse(vnt):
    def result(target_picks, k=5, allow_neg=True):
        target_picks = target_picks[0]
        target_picks = tuple([slice(None, None) if x == None else x for x in target_picks])
        sliced = vnt.tensor.__getitem__(target_picks)
        sliced = jnp.array(sliced)
        sl_flat = sliced.flatten()
        print(sliced.shape)
        stime = time.time()
        vnt_order = jnp.flip(jnp.argsort(sl_flat), axis=0)
        print("sort took", time.time() - stime)
        vnt_topk = vnt_order[:k]
        print(sl_flat[vnt_topk] != 0)
        if not allow_neg:
            vnt_topk = vnt_topk[sl_flat[vnt_topk] != 0]
        print(vnt_topk)
        vnt_topk_indices = [
            x.tolist() for x in jnp.transpose(jnp.array(jnp.unravel_index(vnt_topk, sliced.shape)), [1, 0])
        ]
        vnt_topk_values = sliced.flatten()[
            jnp.array(vnt_topk),
        ].tolist()
        return {"idxs": vnt_topk_indices, "values": vnt_topk_values}

    return result
