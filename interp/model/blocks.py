import math
from typing import Any, Callable, Optional, Tuple, TypeVar, Union

from attrs import frozen
from flax.core.scope import FrozenVariableDict
import flax.linen as nn
import jax
from jax import lax
from jax.nn import initializers
from flax.linen.initializers import lecun_normal, zeros
from flax.linen.normalization import Dtype, PRNGKey, Shape, Array, _compute_stats
import jax.numpy as jnp
from interp.tools.jax_util import stack_tree

from interp.tools.variable_dict import variable_dict_replace
from interp.tools.log import MutLogCache, ShapeLogger
import interp.tools.optional as op


def gelu(x: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jnp.power(x, 3.0))))


class PointwiseNonlinearity(nn.Module):
    # union because mypy thinks all callable members are methods, which this isn't
    nonlin: Union[Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]

    def __call__(self, x: jnp.ndarray, log: MutLogCache) -> jnp.ndarray:  # type: ignore[override]
        id_name = "identity"
        non_lin_name = "nonlinearity_mul"
        if log.log_info.would_log_or_modify(id_name) or log.log_info.would_log_or_modify(non_lin_name):
            id_x = log.log_and_modify(x, id_name)
            nonlinearity_mul = log.log_and_modify(jnp.where(jnp.abs(x) == 0, 0.0, self.nonlin(x) / x), non_lin_name)
            return id_x * nonlinearity_mul
        else:
            return self.nonlin(x)


# (taken from flax and added logging)
class LayerNorm(nn.Module):
    """Layer normalization (https://arxiv.org/abs/1607.06450).
    Operates on the last axis of the input data.

    It normalizes the activations of the layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within
    each example close to 0 and the activation standard deviation close to 1.

    Attributes:
      epsilon: A small float added to variance to avoid dividing by zero.
      dtype: the dtype of the computation (default: float32).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      use_bias:  If True, bias (beta) is added.
      use_scale: If True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: Initializer for bias, by default, zero.
      scale_init: Initializer for scale, by default, one.
    """

    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones

    @nn.compact
    def __call__(self, x: jnp.ndarray, log: Optional[MutLogCache] = None) -> jnp.ndarray:  # type: ignore[override]
        """Applies layer normalization on the input.

        Args:
          x: the inputs

        Returns:
          Normalized inputs (the same shape as inputs).
        """
        log_v = op.unwrap_or(log, MutLogCache.noop())

        reduction_axes = (-1,)
        feature_axes = (-1,)

        mean, var = _compute_stats(x, reduction_axes, None, None)

        stats_shape = list(x.shape)
        for axis in reduction_axes:
            stats_shape[axis] = 1
        mean = log_v.log_and_modify(mean.reshape(stats_shape), "mean")
        var = log_v.log_and_modify(var.reshape(stats_shape), "var")
        feature_shape = [1] * x.ndim
        reduced_feature_shape = []
        for ax in feature_axes:
            feature_shape[ax] = x.shape[ax]
            reduced_feature_shape.append(x.shape[ax])

        y = log_v.log_and_modify(x - mean, "sub_mean")

        if self.use_scale:
            scale = self.param("scale", self.scale_init, reduced_feature_shape, self.param_dtype).reshape(feature_shape)
            y = log_v.log_and_modify(y * scale, "mul_scale")

        overall_mul = log_v.log_and_modify(lax.rsqrt(var + self.epsilon), "overall_mul")
        y = log_v.log_and_modify(y * overall_mul, "multiplied_by_overall")

        if self.use_bias:
            bias = log_v.log_and_modify(
                self.param("bias", self.bias_init, reduced_feature_shape, self.param_dtype).reshape(feature_shape),
                "bias",
            )
            y = log_v.log_and_modify(y + bias, "added_bias")

        return jnp.asarray(y, self.dtype)


class Dense(nn.Module):
    """A linear transformation applied over the last dimension of the input.

    Attributes:
      features: the number of output features.
      use_bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: float32).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
    """

    features: int
    use_bias: bool = True
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    precision: Any = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = lecun_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros

    @nn.compact
    def __call__(self, inputs: Array, log: Optional[MutLogCache] = None) -> Array:  # type: ignore[override]
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        log_v = op.unwrap_or(log, MutLogCache.noop())

        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param("kernel", self.kernel_init, (inputs.shape[-1], self.features), self.param_dtype)
        kernel = jnp.asarray(kernel, self.dtype)
        y = log_v.log_and_modify(
            lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())), precision=self.precision),
            "mul_kernel",
        )
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.param_dtype)
            bias = log_v.log_and_modify(jnp.asarray(bias, self.dtype), "bias")
            y = log_v.log_and_modify(y + jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,)), "added_bias")
        return y

    # we'll use pytorch shape convention when accessing weights externally, so we transpose
    def get_weights(self):
        return jnp.transpose(self.variables["params"]["kernel"])

    def get_bias(self):
        return self.variables["params"]["bias"]

    def replace_weights(self, get_new_weights: Callable[[jnp.ndarray], jnp.ndarray]) -> FrozenVariableDict:
        return variable_dict_replace(
            self.variables, jnp.transpose(get_new_weights(self.get_weights())), "params", "kernel"
        )

    def replace_bias(self, get_new_bias: Callable[[jnp.ndarray], jnp.ndarray]) -> FrozenVariableDict:
        return variable_dict_replace(self.variables, get_new_bias(self.get_bias()), "params", "bias")

    def set_weights(self, new_weights: jnp.ndarray) -> FrozenVariableDict:
        return self.replace_weights(lambda _: new_weights)

    def set_bias(self, new_bias: jnp.ndarray) -> FrozenVariableDict:
        return self.replace_bias(lambda _: new_bias)


@frozen
class ScanRunOnLogConfig:
    use_for_loop: bool = False
    unroll: Union[int, bool] = 1


Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


def scan_run_on_log(
    run_on_log: Callable[[MutLogCache, Carry, X], Tuple[Carry, Y]],
    init: Carry,
    get_x: Callable[[Union[int, jnp.ndarray]], X],
    n: int,
    log: MutLogCache,
    init_idxed: bool = True,
    config: ScanRunOnLogConfig = ScanRunOnLogConfig(),
) -> Tuple[Carry, Y]:
    """
    Motivated by better jax compilation for functions running over sequential layers in the model,
    compared to the for-loop implimentation.

    In the most basic use case (running a forward pass of GPT), Carry is the class of embedding tensor,
    and X is the GptBlock class.

    Helpful context in the jax docs: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
    """

    if init_idxed:
        assert log.log_info.log_idx is None, "would need to be handled separately"

        def get_shape():
            shape_log = MutLogCache.new(logger=ShapeLogger(), log_prefix=log.log_info.log_prefix)
            run_on_log(shape_log.sub(log_idx=jnp.array(0)), init, get_x(0))
            return shape_log.cache

        shapes = jax.eval_shape(get_shape)

        log.init_idxed_for_shapes(shapes, count=n)

    if config.use_for_loop:
        carry: Carry = init
        ys_lst = []
        for i in range(n):
            carry, y = run_on_log(log.sub(log_idx=i), carry, get_x(i))
            ys_lst.append(y)
        ys: Y = stack_tree(ys_lst)
    else:

        def body(carry_cache, i):
            carry, cache = carry_cache
            sub_log = MutLogCache.new_info(log.log_info.sub(log_idx=i), cache)
            new_carry, y = run_on_log(sub_log, carry, get_x(i))
            return (new_carry, sub_log.cache), y

        if isinstance(config.unroll, bool):
            unroll = n if config.unroll else 1
        else:
            unroll = config.unroll

        (carry, log.cache), ys = jax.lax.scan(body, (init, log.cache), jnp.arange(n), unroll=unroll)

    return carry, ys
