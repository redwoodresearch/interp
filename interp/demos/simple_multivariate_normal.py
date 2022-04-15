import jax
import jax.numpy as jnp

from interp.tools.multivariate_normal import MultivariateNormal
from interp.tools.immutable_dict import assign, assign_f, keep_only_f, operate, operate_f

# %%

"""
We can construct multivariate normal distributions by supplying a mean and a
covariance matrix.
"""

# %%

x = MultivariateNormal(jnp.array([1, 2.0]), jnp.array([[1, 0.1], [0.1, 3.0]]))

print(x.mean_as())
print(x.covariance_as())

# %%

"""
Linear functions can be applied.
"""

# %%

mul_3 = x.lin_op(lambda x: x * 3)

print(mul_3.mean_as())
print(mul_3.covariance_as())

# %%

mul_mat = x.lin_op(lambda x: jnp.array([[-1.0, 3.0], [0.01, 0.02]]) @ x)

print(mul_mat.mean_as())
print(mul_mat.covariance_as())

# %%

"""
We can also transform the type and shape of the data.
"""

# %%

x_dict = x.lin_op(lambda x: {"a": x[0].reshape(1, 1, 1), "b": x[1]})

print(x_dict.mean_as())
print(x_dict.covariance_as())

# %%

"""
assign is from interp.tools.immutable_dict which contains various functions for
operating on dictionaries in a more functional way.

The more functional approach is often nicer for working with multivariate
normals which hold various tensors in a dictionary.

(Detail: Also, jax doesn't like it if you mutate the tree def of things in a few
cases, so it's good practice - despite being inefficient - to always copy
before mutation. That's how these functions are immutable.)
"""

# %%

y = x_dict.lin_op(lambda x: assign(x, "c", jnp.array([[x["b"], -x["b"]], [3.0 * x["b"], x["b"]]])))

print(y.mean_as())
print(y.covariance_as())

# %%

"""
This covariance is a bit of a mess, so let's sample to get a sense of what's going on instead.
"""

# %%

samples = y.sample(jax.random.PRNGKey(2), (4,))
samples.shape

# %%

"""
Samples are returned in flattened shape, but we can transform into the data
shape if desired.
"""

# %%

samples_dict = y.flat_value_config.as_tree(samples, dim=-1)
samples_dict

# %%

"""
The details of maintaining a non-flat representation are handled by the `flat_value_config`.
"""

# %%

"""
Other operations are also supported of course.
"""

# %%

add_vals = x_dict.add(lambda x: operate(x, "a", "a", lambda l: l + 4))
print(x_dict.mean_as())
print(add_vals.mean_as())
print(x_dict.covariance_as()["a"])
print(add_vals.covariance_as()["a"])

# %%

set_vals = x_dict.set(
    set_to=2.7, setter=lambda x, set_to: operate(x, "a", "a", lambda x: jnp.broadcast_to(set_to, x.shape))
)
print(x_dict.mean_as())
print(set_vals.mean_as())
print(x_dict.covariance_as()["a"])
print(set_vals.covariance_as()["a"])

# %%

set_vals = x_dict.set(
    # the set to value can be an arbitrary pytree
    set_to={"first": 2.7, "second": 9.1},
    setter=lambda x, set_to: (
        operate_f("a", "a", lambda v: jnp.broadcast_to(set_to["first"], v.shape))
        @ operate_f("b", "b", lambda v: jnp.broadcast_to(set_to["second"], v.shape))
    )(x),
)
print(x_dict.mean_as())
print(set_vals.mean_as())
print(x_dict.covariance_as())
print(set_vals.covariance_as())


# %%

"""
Note that if functions passed to various operations don't do 'what they're
supposed to do', then you'll get incorrect results. For instance, don't pass a
non-linear function to .lin_op (affine isn't ok - just linear!).

Functions with such demands have docstrings explaining what's up.
"""

# %%

# a : (2,)
# b : (3,2)
vals = MultivariateNormal(jnp.ones((8,)), jnp.eye(8)).lin_op(lambda x: {"a": x[:2], "b": x[2:].reshape(3, 2)})

a, b = vals.mean_as()["a"], vals.mean_as()["b"]
jnp.einsum(a, ["n"], b, ["m", "n"], ["m"])

# %%

# compare to einsum above
mulled = vals.mul_select(
    selector_l=lambda x: x["a"],
    l_axes_names=["n"],
    selector_r=lambda x: x["b"],
    r_axes_names=["m", "n"],
    out_axes_names=["m"],
    combine=lambda new, _: new,
)
mulled.covariance_as()

# %%

"""
A decent number of examples for various operations and immutable_dict
operations can be found by going through the apply_to_normal function in
UnidirectionalAttn.
"""


# %%

"""
Another operation of interest is conditioning.
"""

# %%

to_cond_x = MultivariateNormal(jnp.array([1, 2.0]), jnp.array([[1, 0.8], [0.8, 3.0]])).lin_op(
    lambda x: {"a": x[0], "b": x[1]}
)
cond_x = to_cond_x.condition(selector=lambda x: x["a"], value=0.0)
print(cond_x.mean_as())
print(cond_x.covariance_as())
print(cond_x.sample(jax.random.PRNGKey(2838), (10,)))

# %%

"""
We can condition on a == b by setting the difference equal to zero.

Numerics on this aren't great...
"""

# %%

eq_cond_x = to_cond_x.condition(lambda x: x["a"] - x["b"], 0.0)
print(eq_cond_x.sample(jax.random.PRNGKey(2838), (10,)))

# %%

"""
We can also select and condition on a pytree if desired.
"""

# %%


to_cond_x = MultivariateNormal(jnp.zeros((3,)), jnp.eye(3)).lin_op(
    lambda x: {
        "a": x[0] + 3 * x[2],
        "b": x[0] + x[1] * 2 - x[2],
        "c": x[2],
    }
)
cond_x = to_cond_x.condition(
    selector=keep_only_f(["b", "c"]),
    value={"b": -10.0, "c": 12.0},
    # Passing a setter will increase numerical accuracy when we just select a
    # subset - try removing the setter.
    setter=lambda x, set_to: (
        assign_f("b", set_to["b"], check_present=True) @ assign_f("c", set_to["c"], check_present=True)
    )(x),
)
print(cond_x.mean_as())
print(cond_x.covariance_as())
print(cond_x.sample(jax.random.PRNGKey(2838), (10,)))
