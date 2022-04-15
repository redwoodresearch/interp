import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

from interp.tools.test_multivariate_normal import random_normal

key = jax.random.PRNGKey(8)

key, subkey = jax.random.split(key)
normal = random_normal(subkey, 3)

pdf = lambda x: multivariate_normal.pdf(x, normal.mean, normal.covariance)

off_mean = normal.mean - 1e-3

fst_derivs = jax.jacrev(pdf)(off_mean)
snd_derivs = jax.jacrev(jax.jacrev(pdf))(normal.mean)
snd_derivs_off_mean = jax.jacrev(jax.jacrev(pdf))(off_mean)

inv_cov = jnp.linalg.inv(normal.covariance)


def should_be_pdf(x):
    return jnp.exp(-jnp.einsum("m, n, m n", (x - normal.mean), x - normal.mean, inv_cov) / 2) / jnp.sqrt(
        jnp.linalg.det(2 * jnp.pi * normal.covariance)
    )


print("pdf, mean")
print(pdf(normal.mean) - should_be_pdf(normal.mean))
print("pdf, off mean")
print(pdf(off_mean) - should_be_pdf(off_mean))


def should_be_snd_deriv(x):
    return -should_be_pdf(x) * (inv_cov - jnp.einsum("n m, n, m -> n m", inv_cov, x - normal.mean, x - normal.mean))


print("snd, mean")
print(should_be_snd_deriv(normal.mean))
print(snd_derivs)
print(snd_derivs - should_be_snd_deriv(normal.mean))

print("snd, off mean")
print(should_be_snd_deriv(off_mean))
print(snd_derivs_off_mean)
print(snd_derivs_off_mean - should_be_snd_deriv(off_mean))

print("revived")
revived_cov = jnp.linalg.inv(snd_derivs_off_mean / (-should_be_pdf(off_mean)))
print(revived_cov)
print(revived_cov - normal.covariance)


def should_be_fst_deriv_at(x):
    return -should_be_pdf(x) * jnp.einsum("n m, m -> n", inv_cov, x - normal.mean)


print("fst, off mean")
print(should_be_fst_deriv_at(off_mean))
print(fst_derivs)
print(should_be_fst_deriv_at(off_mean) - fst_derivs)
