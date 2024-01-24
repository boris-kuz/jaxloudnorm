import warnings
import jax.numpy as jnp
import jax
import numpy as np

from . import util
from .iirfilter import IIRfilter


class Meter(object):
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

    def __init__(self, rate, filter_class="K-weighting", block_size=0.400):
        self.rate = rate
        self.filter_class = filter_class
        self.block_size = block_size

    def integrated_loudness(self, data):
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
        input_data = data
        util.valid_audio(input_data, self.rate, self.block_size)

        if input_data.ndim == 1:
            input_data = jnp.reshape(input_data, (input_data.shape[0], 1))

        numChannels = input_data.shape[1]
        numSamples = input_data.shape[0]

        input_data = jnp.asarray(input_data)

        # Apply frequency weighting filters - account for the acoustic response of the head and auditory system
        for filter_stage in self._filters.values():
            input_data = filter_stage.apply_filter(input_data, axis=0)

        G = jnp.asarray([1.0, 1.0, 1.0, 1.41, 1.41])  # channel gains
        T_g = self.block_size  # 400 ms gating block standard
        Gamma_a = -70.0  # -70 LKFS = absolute loudness threshold
        overlap = 0.75  # overlap of 75% of the block duration
        step = 1.0 - overlap  # step size by percentage

        T = numSamples / self.rate  # length of the input in seconds
        numBlocks = int(
            round(((T - T_g) / (T_g * step))) + 1
        )  # total number of gated blocks (see end of eq. 3)
        z = jnp.zeros(
            shape=(numChannels, numBlocks)
        )  # instantiate array - trasponse of input

        j_range = list(range(0, numBlocks))  # indexed list of total blocks
        indices = [
            (int(T_g * (j * step) * self.rate), int(T_g * (j * step + 1) * self.rate))
            for j in j_range
        ]
        input_slices = jnp.asarray([input_data[l:u, ...] for l, u in indices])
        z = (jnp.reciprocal(T_g * self.rate)) * jnp.sum(
            jnp.square(input_slices), axis=1
        ).T

        # loudness for each jth block (see eq. 4)
        loudness_per_block = -0.691 + 10.0 * jnp.log10(
            jnp.sum(G[:numChannels] * z[:numChannels, ...], axis=0)
        )

        # find gating block indices above absolute threshold
        # abs_gating_idxs = [
        #     idx
        #     for idx, loudness in enumerate(loudness_per_block)
        #     if loudness >= Gamma_a
        # ]

        # calculate the average of z[i,j] as show in eq. 5
        # z_avg_gated = jnp.mean(z[:, abs_gating_idxs], axis=1)

        z_ = jax.lax.select(
            jnp.tile((loudness_per_block >= Gamma_a).reshape(1, -1), numChannels),
            z[:numChannels, ...],
            jnp.zeros_like(z[:numChannels, ...]),
        )
        not_zero = jnp.count_nonzero(z_, axis=1)
        z_avg_gated = jnp.sum(z_, axis=1) / not_zero

        # calculate the relative threshold value (see eq. 6)
        Gamma_r = (
            -0.691
            + 10.0 * jnp.log10(jnp.sum(G[:numChannels] * z_avg_gated[:numChannels]))
            - 10.0
        )

        # find gating block indices above relative and absolute thresholds  (end of eq. 7)
        # abs_and_rel_gating_idxs = [
        #     j
        #     for j, l_j in enumerate(loudness_per_block)
        #     if (l_j > Gamma_r and l_j > Gamma_a)
        # ]

        # # calculate the average of z[i,j] as show in eq. 7 with blocks above both thresholds
        # z_avg_gated = jnp.nan_to_num(jnp.mean(z[:, abs_and_rel_gating_idxs], axis=1))

        z_ = jax.lax.select(
            jnp.logical_and(
                jnp.tile((loudness_per_block >= Gamma_a).reshape(1, -1), numChannels),
                jnp.tile((loudness_per_block >= Gamma_r).reshape(1, -1), numChannels),
            ),
            z[:numChannels, ...],
            jnp.zeros_like(z[:numChannels, ...]),
        )
        not_zero = jnp.count_nonzero(z_, axis=1)
        z_avg_gated = jnp.nan_to_num(jnp.sum(z_, axis=1) / not_zero)

        # calculate final loudness gated loudness (see eq. 7)
        with np.errstate(divide="ignore"):
            LUFS = -0.691 + 10.0 * jnp.log10(
                jnp.sum(G[:numChannels] * z_avg_gated[:numChannels])
            )

        return LUFS

    @property
    def filter_class(self):
        return self._filter_class

    @filter_class.setter
    def filter_class(self, value):
        self._filters = {}  # reset (clear) filters
        self._filter_class = value
        if self._filter_class == "K-weighting":
            self._filters["high_shelf"] = IIRfilter(
                4.0, 1 / jnp.sqrt(2), 1500.0, self.rate, "high_shelf"
            )
            self._filters["high_pass"] = IIRfilter(
                0.0, 0.5, 38.0, self.rate, "high_pass"
            )
        elif self._filter_class == "Fenton/Lee 1":
            self._filters["high_shelf"] = IIRfilter(
                5.0, 1 / jnp.sqrt(2), 1500.0, self.rate, "high_shelf"
            )
            self._filters["high_pass"] = IIRfilter(
                0.0, 0.5, 130.0, self.rate, "high_pass"
            )
            self._filters["peaking"] = IIRfilter(
                0.0, 1 / jnp.sqrt(2), 500.0, self.rate, "peaking"
            )
        elif self._filter_class == "Fenton/Lee 2":  # not yet implemented
            self._filters["high_self"] = IIRfilter(
                4.0, 1 / jnp.sqrt(2), 1500.0, self.rate, "high_shelf"
            )
            self._filters["high_pass"] = IIRfilter(
                0.0, 0.5, 38.0, self.rate, "high_pass"
            )
        elif self._filter_class == "Dash et al.":
            self._filters["high_pass"] = IIRfilter(
                0.0, 0.375, 149.0, self.rate, "high_pass"
            )
            self._filters["peaking"] = IIRfilter(
                -2.93820927, 1.68878655, 1000.0, self.rate, "peaking"
            )
        elif self._filter_class == "DeMan":
            self._filters["high_shelf_DeMan"] = IIRfilter(
                3.99984385397,
                0.7071752369554193,
                1681.9744509555319,
                self.rate,
                "high_shelf_DeMan",
            )
            self._filters["high_pass_DeMan"] = IIRfilter(
                0.0, 0.5003270373253953, 38.13547087613982, self.rate, "high_pass_DeMan"
            )
        elif self._filter_class == "custom":
            pass
        else:
            raise ValueError("Invalid filter class:", self._filter_class)
