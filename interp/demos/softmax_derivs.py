import jax
import jax.numpy as jnp

from interp.tools.custom_jvp import ablation_custom_jvp, integrated_gradients_custom_jvp

base = jnp.array([[1.0, 2.0, 5, 4], [1.0, 2.0, 5, 4]])
multipliers = jnp.array([[2.0, -1.0, 1.2, 1.9], [1.0, 10.0, 0.002, -1.4]])


def get_full(x):
    return base + jnp.expand_dims(jnp.einsum("n, n m -> m", x, multipliers), range(base.ndim - 1))


for is_log_softmax in [False, True]:
    print(f'for {"log softmax" if is_log_softmax else "softmax"}')
    func = jax.nn.log_softmax if is_log_softmax else jax.nn.softmax

    def get_example_func(func):
        def f(x):
            return func(get_full(x))

        return f

    example_func_actual = get_example_func(func)

    inps = jnp.array([1.0, 1.0])
    actual_deriv = jax.jacfwd(example_func_actual)(inps)
    print("actual:", actual_deriv)
    ablation = jnp.expand_dims(example_func_actual(inps), -1) - jnp.stack(
        [example_func_actual(inps.at[0].mul(0.0)), example_func_actual(inps.at[1].mul(0.0))],
        axis=-1,
    )
    print("ablation:", ablation)
    print("ratio:", actual_deriv / ablation)

    ig_func = integrated_gradients_custom_jvp(func)  # type: ignore[arg-type]
    example_func_ig = get_example_func(ig_func)

    ig_deriv = jax.jacfwd(example_func_ig)(inps)
    print("ig:", ig_deriv)
    assert jnp.allclose(jax.jacrev(example_func_ig)(inps), ig_deriv, atol=3e-3)  # hmmm

    ablation_func = ablation_custom_jvp(func)  # type: ignore[arg-type]
    example_func_ablation = get_example_func(ablation_func)

    ablation_deriv = jax.jacfwd(example_func_ablation)(inps)
    print("ablation wrapper:", ablation_deriv)
    assert jnp.allclose(ablation_deriv, ablation)
