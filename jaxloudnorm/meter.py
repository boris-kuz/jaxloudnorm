import jax.numpy as jnp
import jax
import numpy as np

from jaxtyping import Float, Array, Scalar
from typing import Union

from . import util
from .iirfilter import IIRfilter

from functools import partial


def _gated_mean(condition, x):
    masked = jax.lax.select(
        jnp.broadcast_to(jnp.atleast_2d(condition), x.shape),
        x,
        jnp.zeros_like(x),
    )
    not_zero = jnp.count_nonzero(masked, axis=1)
    return jnp.sum(masked, axis=1) / not_zero


class Meter:
    """Meter object which defines how the meter operates

    Defaults to the algorithm defined in ITU-R BS.1770-4.

    Parameters
    ----------
    rate : float
        Sampling rate in Hz.
    filter_class : str
        Class of weighting filter used.
        - 'K-weighting'
        - 'Fenton/Lee 1'
        - 'Fenton/Lee 2'
        - 'Dash et al.'
        - 'DeMan'
    block_size : float
        Gating block size in seconds.
    """

    def __init__(
        self, rate: float, filter_class: str = "K-weighting", block_size: float = 0.400
    ):
        self.rate = rate
        self.filter_class = filter_class
        self.block_size = block_size

    def integrated_loudness(
        self, input_data: Union[Float[Array, "time channels"], Float[Array, "time"]]
    ) -> Scalar:
        """Measure the integrated gated loudness of a signal.

        Uses the weighting filters and block size defined by the meter
        the integrated loudness is measured based upon the gating algorithm
        defined in the ITU-R BS.1770-4 specification.

        Input data must have shape (samples, ch) or (samples,) for mono audio.
        Supports up to 5 channels and follows the channel ordering:
        [Left, Right, Center, Left surround, Right surround]

        Params
        -------
        data : ndarray
            Input multichannel audio data.

        Returns
        -------
        LUFS : float
            Integrated gated loudness of the input measured in dB LUFS.
        """
        util.valid_audio(input_data, self.rate, self.block_size)

        input_data = jnp.atleast_2d(jnp.asarray(input_data).T).T

        num_samples, num_channels = input_data.shape

        # Apply frequency weighting filters - account for the acoustic response of the head and auditory system
        for filter_stage in self._filters:
            input_data = filter_stage.apply_filter(input_data, axis=0)

        g = jnp.array([1.0, 1.0, 1.0, 1.41, 1.41])  # channel gains
        gating_len_s = self.block_size  # 400 ms gating block standard
        abs_loudness_thresh = -70.0  # -70 LKFS = absolute loudness threshold
        overlap = 0.75  # overlap of 75% of the block duration
        step = 1.0 - overlap  # step size by percentage

        signal_len_s = num_samples / self.rate
        num_gated_blocks = int(
            round(((signal_len_s - gating_len_s) / (gating_len_s * step))) + 1
        )

        indices = [
            tuple(
                int(gating_len_s * x * self.rate)
                for x in (block * step, block * step + 1)
            )
            for block in range(0, num_gated_blocks)
        ]
        slice_max_len = indices[0][1] - indices[0][0]
        input_slices = jnp.asarray(
            [
                jnp.zeros((slice_max_len, input_data.shape[1]))
                .at[: input_data[l:u, ...].shape[0], ...]
                .set(input_data[l:u, ...])
                for l, u in indices
            ]
        )
        z = (
            jnp.reciprocal(gating_len_s * self.rate)
            * jnp.sum(jnp.square(input_slices), axis=1).T
        )

        # loudness for each jth block (see eq. 4)
        loudness_per_block = -0.691 + 10.0 * jnp.log10(
            jnp.sum(g[:num_channels, None] * z[:num_channels, ...], axis=0)
        )

        # find gating block indices above absolute threshold

        z_avg_gated_abs = _gated_mean(
            loudness_per_block >= abs_loudness_thresh, z[:num_channels, ...]
        )

        # calculate the relative threshold value (see eq. 6)
        rel_loudness_thresh = (
            -0.691
            + 10.0
            * jnp.log10(jnp.sum(g[:num_channels] * z_avg_gated_abs[:num_channels]))
            - 10.0
        )

        # find gating block indices above relative and absolute thresholds  (end of eq. 7)
        z_avg_gated_abs_rel = _gated_mean(
            jnp.logical_and(
                loudness_per_block >= abs_loudness_thresh,
                loudness_per_block >= rel_loudness_thresh,
            ),
            z[:num_channels, ...],
        )

        # calculate final loudness gated loudness (see eq. 7)
        return -0.691 + 10.0 * jnp.log10(
            jnp.sum(g[:num_channels] * z_avg_gated_abs_rel[:num_channels])
        )

    @property
    def filter_class(self):
        return self._filter_class

    @filter_class.setter
    def filter_class(self, value):
        make_filter = lambda G, Q, fc, filter_type: IIRfilter(
            G, Q, fc, self.rate, filter_type
        )
        self._filter_class = value
        if self._filter_class == "K-weighting":
            self._filters = [
                make_filter(4.0, 1 / jnp.sqrt(2), 1500.0, "high_shelf"),
                make_filter(0.0, 0.5, 38.0, "high_pass"),
            ]
        elif self._filter_class == "Fenton/Lee 1":
            self._filters = [
                make_filter(5.0, 1 / jnp.sqrt(2), 1500.0, "high_shelf"),
                make_filter(0.0, 0.5, 130.0, "high_pass"),
                make_filter(0.0, 1 / jnp.sqrt(2), 500.0, "peaking"),
            ]
        elif self._filter_class == "Fenton/Lee 2":  # not yet implemented
            self._filters = [
                make_filter(4.0, 1 / jnp.sqrt(2), 1500.0, "high_shelf"),
                make_filter(0.0, 0.5, 38.0, "high_pass"),
            ]
        elif self._filter_class == "Dash et al.":
            self._filters = [
                make_filter(0.0, 0.375, 149.0, "high_pass"),
                make_filter(-2.93820927, 1.68878655, 1000.0, "peaking"),
            ]
        elif self._filter_class == "DeMan":
            self._filters = [
                make_filter(
                    3.99984385397,
                    0.7071752369554193,
                    1681.9744509555319,
                    "high_shelf_DeMan",
                ),
                make_filter(
                    0.0,
                    0.5003270373253953,
                    38.13547087613982,
                    "high_pass_DeMan",
                ),
            ]
        elif self._filter_class == "custom":
            pass
        else:
            raise ValueError("Invalid filter class:", self._filter_class)
