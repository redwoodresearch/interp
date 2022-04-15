import jax.numpy as jnp

import scipy.stats as stats
import numpy as np
import jax.scipy.stats as jstats

from interp.tools.indexer import I
from interp.model.gpt_model import Gpt
from interp.tools.log import LogCache, KeyIdxs, Idxs
from interp.model.gpt_modules import UnidirectionalAttn


class CompareActivation:
    """
    Kinda hacked together class for comparing activations from dist prop to monte carlo values.

    Just does attn activations atm.
    """

    def __init__(
        self,
        model: Gpt,
        monte_log: LogCache,
        dist_log: LogCache,
        monte_log_name: str,
        dist_log_name: str,
        dist_item_name: str,
        seq_axis: int,
        remove_upper_triangle=False,
    ):
        self.seq_axis = seq_axis
        self.monte_stacked = (
            monte_log.get(KeyIdxs(f"blocks.attention.{monte_log_name}", Idxs.all()))
            .swapaxes(0, 1)
            .swapaxes(2, seq_axis + 2)
            .squeeze(axis=2)
        )

        self.monte_mean = self.monte_stacked.mean(axis=0)
        self.monte_var = self.monte_stacked.var(axis=0)
        # cut first to match monte (just begin token)
        dist_idx = (0,) + (I[:],) * (seq_axis - 1) + (I[1:],) + (I[:],) * (self.monte_stacked.ndim - 2 - seq_axis)

        # TODO?
        self.dist_mean = jnp.stack(
            [
                dist_log.get(KeyIdxs.single(f"blocks.attention.{monte_log_name}", i)).mean_as()[dist_item_name][
                    dist_idx
                ]
                for i in range(model.num_layers)
            ]
        )

        self.diff_mean = self.dist_mean - self.monte_mean

        self.norm_stat, _ = stats.normaltest(np.array(self.monte_stacked))

        flat_indexes_base = jnp.argsort(self.norm_stat, axis=None)
        non_flat_base = jnp.unravel_index(flat_indexes_base, self.norm_stat.shape)
        if remove_upper_triangle:
            self.norm_stat_mask = UnidirectionalAttn.mask_out_upper_triangle(
                self.norm_stat, d=0.0, l_idxs=jnp.arange(1, self.monte_mean.shape[seq_axis] + 1)
            ).mean(axis=(-2, -1))
            self.flat_indexes = flat_indexes_base[(non_flat_base[2] >= non_flat_base[3]) & (non_flat_base[2] > 0)]
        else:
            self.flat_indexes = flat_indexes_base

        self.non_flat_idxs = jnp.unravel_index(self.flat_indexes, self.norm_stat.shape)
        self.flat_indexes_np = np.array(self.flat_indexes)

        self.dist_cov = jnp.stack(
            [
                dist_log.get(KeyIdxs.single(f"blocks.attention.{monte_log_name}", i)).covariance_as()[dist_item_name][
                    dist_item_name
                ][dist_idx * 2]
                for i in range(model.num_layers)
            ]
        )

    # TODO: plot direction instead of just std basis
    def plot_layer_idx(self, layer, idx, bin_size=0.1, is_orig_idx=True):
        idx = list(idx)
        if is_orig_idx:
            idx[self.seq_axis - 1] -= 1
        idx = tuple(idx)

        idx_dist_var = self.dist_cov[(layer,) + idx + idx]
        idx_dist_std = jnp.sqrt(idx_dist_var)
        idx_dist_mean = self.dist_mean[(layer,) + idx]

        idx_monte_std = jnp.sqrt(self.monte_var[(layer,) + idx])
        idx_monte_mean = self.monte_mean[(layer,) + idx]

        x_pdf = jnp.linspace(idx_dist_mean - 3 * idx_dist_std, idx_dist_mean + 3 * idx_dist_std)
        y_pdf = jstats.norm.pdf(x_pdf, loc=idx_dist_mean, scale=idx_dist_std)

        x_monte_pdf = jnp.linspace(idx_monte_mean - 3 * idx_monte_std, idx_monte_mean + 3 * idx_monte_std)
        y_monte_pdf = jstats.norm.pdf(x_pdf, loc=idx_monte_mean, scale=idx_monte_std)

        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                name="monte",
                x=self.monte_stacked[(I[:], layer) + idx],
                histnorm="probability",
                xbins=go.histogram.XBins(size=bin_size),
            )
        )
        fig.add_trace(go.Scatter(name="dist pdf", x=x_pdf, y=y_pdf * bin_size, mode="lines"))
        fig.add_trace(go.Scatter(name="monte pdf", x=x_monte_pdf, y=y_monte_pdf * bin_size, mode="lines"))

        return fig

    def plot_percentile(self, percentile: float, bin_size=0.1):
        layer, *idx = np.unravel_index(
            self.flat_indexes_np[min(int(percentile * self.flat_indexes_np.size), self.flat_indexes_np.size - 1)],
            self.norm_stat.shape,
        )

        return self.plot_layer_idx(layer, idx, bin_size=bin_size, is_orig_idx=False)
