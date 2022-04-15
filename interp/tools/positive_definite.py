from jax.numpy import linalg as la
import jax.numpy as jnp
import jax

# Allow for exact symmetry tests. A bit silly, but whatever...
# maybe should instead set lower triangle equal to upper triangle
@jax.jit
def make_sym(x: jnp.ndarray):
    return jnp.maximum(x, jnp.transpose(x))


# from https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194
# this is pretty slow, and not currently used for normal propagation
@jax.jit
def nearest_positive_definite_not_nan(
    A,
    # this eps is pretty big, but we're down with some sketchy stuff
    eps=1e-7,
    iter_limit=10,
):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = jnp.dot(V.T, jnp.dot(jnp.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    # spacing modified from stack overflow
    spacing = la.norm(A) * eps

    def cond(x):
        # if we have nans, exit because there is no hope (and maybe we ran out of iters?)
        return (~jnp.isnan(x["A3"]).any()) & (~is_positive_definite(x["A3"]))

    def update_state(x):
        mineig = jnp.min(jnp.real(la.eigvalsh(x["A3"])))
        next_mat = x["A3"] + jnp.eye(A.shape[0]) * (-mineig * x["k"] ** 2 + spacing)
        next_mat = jnp.where(x["k"] > iter_limit, float("nan"), next_mat)

        return dict(A3=next_mat, k=x["k"] + 1)

    return jax.lax.while_loop(cond, update_state, dict(A3=A3, k=jnp.array(1)))["A3"]


@jax.jit
def nearest_positive_definite(
    A,
    eps=1e-7,
    iter_limit=10,
):
    return jax.lax.cond(
        jnp.isnan(A).any(),
        lambda *_: jnp.full_like(A, float("nan"), dtype=A.dtype),
        nearest_positive_definite_not_nan,
        A,
        eps,
        iter_limit,
    )


@jax.jit
def is_positive_definite(B):
    """Returns true when input is positive-definite, via Cholesky"""
    return ~jnp.isnan(la.cholesky(B)).any()


def check_valid_positive_semi_def(A):
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    assert not jnp.isnan(A).any()
    assert (A == jnp.transpose(A)).all()

    def get_info():
        eig_vals = jnp.linalg.eigvalsh(A)
        return f"\neigvals: {eig_vals}\n min eigval: {eig_vals.min()}"

    assert is_positive_definite(A), f"cholesky failed, not pos semi-def" + get_info()
