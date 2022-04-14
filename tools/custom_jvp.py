from typing import Callable, Any

import jax
import jax.numpy as jnp

# only works on one argument functions (for now)
def get_integrated_gradients(f: Callable[[Any], Any], x, x_dot, min_mul=0.0, max_mul=1.0, n=30):
    def mean_func(addr):
        return jax.vmap(f)(
            jnp.expand_dims(jnp.linspace(min_mul, max_mul, num=n, dtype=x.dtype), range(1, x.ndim + 1))
            * jnp.expand_dims(x, 0)
            + jnp.expand_dims(addr, 0)
        ).mean(axis=0)

    return jax.jvp(mean_func, [jnp.zeros_like(x_dot)], [x_dot])[1]


def integrated_gradients_custom_jvp(f: Callable[[Any], Any], min_mul=0.0, max_mul=1.0, n=30):
    out = jax.custom_jvp(lambda x: f(x))
    out.defjvps(lambda x_dot, _, x: get_integrated_gradients(f, x, x_dot, min_mul=min_mul, max_mul=max_mul, n=n))
    return out


def get_mix_with_g_grad(f: Callable[[Any], Any], g: Callable[[Any], Any], x, x_dot, frac_g=0.5):
    def mean_func(addr):
        return f(x + addr) * (1 - frac_g) + g(x + addr) * frac_g

    return jax.jvp(mean_func, [jnp.zeros_like(x_dot)], [x_dot])[1]


def mix_with_g_custom_jvp(f: Callable[[Any], Any], g: Callable[[Any], Any], frac_g=0.5):
    out = jax.custom_jvp(lambda x: f(x))
    out.defjvps(lambda x_dot, _, x: get_mix_with_g_grad(f, g, x, x_dot, frac_g))
    return out


def mix_with_linear_custom_jvp(f: Callable[[Any], Any], frac_linear=0.5):
    return mix_with_g_custom_jvp(f, lambda x: x, frac_linear)


# only works on one argument functions (for now)
# only works on fwd
def get_ablation(f: Callable[[Any], Any], x, x_dot):
    applied_orig = f(jax.lax.stop_gradient(x))
    orig = jax.lax.stop_gradient(x)

    def ablate_all_tangents(x_dot_tan):
        diff = orig - x_dot_tan
        additional_dims = diff.ndim - applied_orig.ndim

        vmapped_func = f
        for _ in range(additional_dims):
            vmapped_func = jax.vmap(vmapped_func)

        applied_diff = vmapped_func(diff)
        assert applied_diff.shape[-applied_orig.ndim :] == applied_orig.shape
        to_set = applied_orig - applied_diff

        # with great hackyness comes great power
        if hasattr(x_dot_tan, "val") and hasattr(x_dot_tan.val, "tangent"):
            to_set.val.tangent = ablate_all_tangents(x_dot_tan.val.tangent)

        return to_set

    return ablate_all_tangents(x_dot)


def ablation_custom_jvp(f: Callable[[Any], Any]):
    out = jax.custom_jvp(lambda x: f(x))
    out.defjvps(lambda x_dot, _, x: get_ablation(f, x, x_dot))
    return out


def different_function_custom_jvp(f, new):
    out = jax.custom_jvp(f)
    out.jvp = lambda primals, tangents: jax.jvp(new, primals, tangents)

    return out
