from typing import Tuple

from flax.core.scope import FrozenVariableDict
import jax
import jax.numpy as jnp
import numpy as np
import plotly.express as px
import einops
from plotly.graph_objects import Figure

from interp.model.gpt_model import Gpt


def canonicalize(A: jnp.ndarray, B: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Canonicalizes A and B by computing the SVD UDW^T = AB^T, and
    computing A' = U*sqrt(D) and B' = W*sqrt(D).

    :param A: tensor of shape (c, e) where c <= e
    :param B: tensor of shape (c, e)
    :return: (A', B', D); A'.shape = B'.shape = (c, e) and D.shape = (c,)
    """
    c, e = A.shape
    assert (c, e) == B.shape, (A.shape, B.shape)
    assert c <= e, (c, e)

    AB = jnp.einsum("ce, cf -> ef", A, B)
    U, D, Wt = jnp.linalg.svd(AB)
    D_nonzero = D[:c]
    assert jnp.allclose(D[c:], jnp.zeros(e - c), atol=1e-3), jnp.max(D[c:])

    A_canon = jnp.einsum("ec, c -> ce", U[:, :c], jnp.sqrt(D_nonzero))
    B_canon = jnp.einsum("cf, c -> cf", Wt[:c, :], jnp.sqrt(D_nonzero))

    AB_canon = jnp.einsum("ce,cf -> ef", A_canon, B_canon)
    assert jnp.allclose(AB_canon, AB, atol=1e-3), jnp.max(jnp.abs(AB_canon - AB))

    return A_canon, B_canon, D_nonzero


def canonicalize_qk(
    model: Gpt, params: FrozenVariableDict, layer: int, head: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    q, k = model.bind(params).get_qk_mats_all_layers()[:, layer, head]
    return canonicalize(q, k)


def canonicalize_ov(
    model: Gpt, params: FrozenVariableDict, layer: int, head: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    v = model.bind(params).get_v_mats_all_layers()[layer, head]
    o = model.bind(params).get_o_mats_all_layers()[layer, head]
    o_canon_T, v_canon, d = canonicalize(o.T, v)
    return jnp.transpose(o_canon_T), v_canon, d


def canonicalize_matrix_product(
    model: Gpt, params: FrozenVariableDict, layer: int, head: int, matrices: str = "qk"
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    assert matrices in ["qk", "ov"], matrices
    fn = canonicalize_qk if matrices == "qk" else canonicalize_ov
    return fn(model, params, layer, head)


def normalize(x, axis=-1, eps=1e-12):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=axis, keepdims=True), eps)


def cosine_similarity(A: jnp.ndarray, B: jnp.ndarray, normalized: bool = True) -> jnp.ndarray:
    """
    Computes cosine similarities of rows of A and B.

    :param A: tensor
    :param B: tensor of the same shape as A
    :param normalized: if True, this function computes cosine similarity. Otherwise it computes dot products of rows of A and B. Default: True.
    :return: C where C[i,j] is the similarity between A[i] and B[j]
    """
    assert A.shape[-1] == B.shape[-1], (A.shape, B.shape)

    if normalized:
        A = normalize(A)
        B = normalize(B)

    prod = jnp.einsum("ce,de -> cd", A, B)
    return prod


def show_cosine_similarity(A: jnp.ndarray, B: jnp.ndarray, normalized: bool = True, rang=(-0.5, 0.5)) -> Figure:
    prod = cosine_similarity(A, B, normalized=normalized)
    fig = px.imshow(np.array(prod), range_color=rang, color_continuous_scale="RdBu")
    fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
    return fig


def cosine_similarities(As: jnp.ndarray, Bs: jnp.ndarray, normalized: bool = True) -> jnp.ndarray:
    """
    Computes cosine similarities of all pairs of matrices A and B.

    :param normalized: if True, this function computes cosine similarity. Otherwise it computes dot products of rows. Default: True.
    """
    if normalized:
        As = normalize(As)
        Bs = normalize(Bs)

    prod = jnp.einsum("nce,mde -> nmcd", As, Bs)
    glommed = einops.rearrange(prod, "n m c d -> (n c) (m d)")
    return glommed


def cosine_similarities_canon(
    model: Gpt, params: FrozenVariableDict, layer: int, matrices: str = "qk", normalized: bool = True
) -> jnp.ndarray:
    """
    Computes cosine similarities of rows between all (canonicalized) matrices in a layer.

    :param matrices: which two matrix types to compare; e.g. "vv" compares all (canonicalized) V matrices in a layer with each other.
    :param normalized: if True, this function computes cosine similarity. Otherwise it computes dot products of rows. Default: True.
    :return: C where C[h1*size + i, h2*size + j] is the similarity between i'th row of matrices[0] in head h1, and j'th row of matrices[1] in h2
    """
    assert matrices in ["qk", "qq", "kq", "kk", "ov", "oo", "vo", "vv"], matrices
    product_to_compute = "qk" if "q" in matrices or "k" in matrices else "ov"

    As_to_stack = []
    Bs_to_stack = []
    for h in range(model.num_heads):
        A, B, _ = canonicalize_matrix_product(model, params, layer, h, product_to_compute)
        As_to_stack.append(A)
        Bs_to_stack.append(B)

    As = jnp.stack(As_to_stack)
    Bs = jnp.stack(Bs_to_stack)

    def matrix(i):
        name = matrices[i]
        idx = product_to_compute.index(name)
        return [As, Bs][idx]

    return cosine_similarities(matrix(0), matrix(1), normalized=normalized)


def show_cosine_similarities_canon(
    model: Gpt, params: FrozenVariableDict, layer: int, matrices: str = "qk", normalized: bool = True, rang=(-0.5, 0.5)
) -> Figure:
    transpose_products = cosine_similarities_canon(model, params, layer, matrices, normalized=normalized)
    fig = px.imshow(
        np.array(transpose_products),
        range_color=rang,
        color_continuous_scale="RdBu",
        labels=dict(x=matrices[0], y=matrices[1]),
    )
    fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
    return fig
