import timeit
import numpy as np
from functools import partial

import jax

import jaxloudnorm as pyln


def test_loudness(
    batch_size=4,
    seconds=30,
    rate=44100,
    channels=2,
):

    @partial(jax.jit, static_argnames='rate')
    def jit_integrated_loudness(data, rate):
        meter = pyln.Meter(rate, block_size=0.400, use_fir=True, zeros=512)
        loudness = jax.vmap(meter.integrated_loudness)(data)
        return loudness

    data = jax.random.uniform(jax.random.key(0),
                              shape=(batch_size, int(rate*seconds), channels),
                              minval=-0.5,
                              maxval=0.5)

    out = jit_integrated_loudness(data, rate).block_until_ready()

    execution_times = timeit.repeat('jit_integrated_loudness(data, rate).block_until_ready()',
                                    number=1, repeat=10,
                                    globals={'data': data, 'rate': rate, 'jit_integrated_loudness': jit_integrated_loudness})
    execution_times = np.array(execution_times) * 1000  # convert to ms
    mean_time = execution_times.mean()
    median_time = np.median(execution_times)
    min_time = execution_times.min()
    max_time = execution_times.max()
    std_time = execution_times.std()

    print(f"Num executions:", execution_times.shape[0])
    print(f"Mean execution time: {mean_time:.2f} ms")
    print(f"Median execution time: {median_time:.2f} ms")
    print(f"Min execution time: {min_time:.2f} ms")
    print(f"Max execution time: {max_time:.2f} ms")
    print(f"Std execution time: {std_time:.2f} ms")
