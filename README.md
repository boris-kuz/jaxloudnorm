`s/np/jnp/g`
`s/py/jax/g`

Docs below are from the [original repo](https://github.com/csteinmetz1/pyloudnorm) for the most part.

# jaxloudnorm  [![Build Status](https://travis-ci.org/csteinmetz1/pyloudnorm.svg?branch=master)](https://travis-ci.org/csteinmetz1/pyloudnorm) ![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.3551801.svg)
Flexible audio loudness meter in Python. 

Implementation of [ITU-R BS.1770-4](https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-4-201510-I!!PDF-E.pdf). <br/>
Allows control over gating block size and frequency weighting filters for additional control. 

For full details on the implementation see our [paper](https://csteinmetz1.github.io/pyloudnorm-eval/paper/pyloudnorm_preprint.pdf) with a summary in our [AES presentation video](https://www.youtube.com/watch?v=krSJpQ3d4gE).

## Installation
Install from GitHub repo using
```
pip install git+https://github.com/boris-kuz/jaxloudnorm
```
## Usage

### Find the loudness of an audio file
It's easy to measure the loudness of a wav file. 
Here we use PySoundFile to read a .wav file as an ndarray.
```python
import soundfile as sf
import jaxloudnorm as jln

data, rate = sf.read("test.wav") # load audio (with shape (samples, channels))
meter = jln.Meter(rate) # create BS.1770 meter
loudness = meter.integrated_loudness(data) # measure loudness
```

### Loudness normalize and peak normalize audio files
Methods are included to normalize audio files to desired peak values or desired loudness.
```python
import soundfile as sf
import jaxloudnorm as jln

data, rate = sf.read("test.wav") # load audio

# peak normalize audio to -1 dB
peak_normalized_audio = jln.normalize.peak(data, -1.0)

# measure the loudness first 
meter = jln.Meter(rate) # create BS.1770 meter
loudness = meter.integrated_loudness(data)

# loudness normalize audio to -12 dB LUFS
loudness_normalized_audio = jln.normalize.loudness(data, loudness, -12.0)
```

### Advanced operation
A number of alternate weighting filters are available, as well as the ability to adjust the analysis block size. 
Examples are shown below.
```python
import soundfile as sf
import jaxloudnorm as jln
from jaxloudnorm import IIRfilter

data, rate = sf.read("test.wav") # load audio

# block size
meter1 = jln.Meter(rate)                               # 400ms block size
meter2 = jln.Meter(rate, block_size=0.200)             # 200ms block size

# filter classes
meter3 = jln.Meter(rate)                               # BS.1770 meter
meter4 = jln.Meter(rate, filter_class="DeMan")         # fully compliant filters  
meter5 = jln.Meter(rate, filter_class="Fenton/Lee 1")  # low complexity improvement by Fenton and Lee
meter6 = jln.Meter(rate, filter_class="Fenton/Lee 2")  # higher complexity improvement by Fenton and Lee
meter7 = jln.Meter(rate, filter_class="Dash et al.")   # early modification option

# create your own IIR filters
my_high_pass  = IIRfilter(0.0, 0.5, 20.0, rate, 'high_pass')
my_high_shelf = IIRfilter(2.0, 0.7, 1525.0, rate, 'high_shelf')

# create a meter initialized without filters
meter8 = jln.Meter(rate, filter_class="custom")

# load your filters into the meter
meter8._filters = [my_high_pass, my_high_shelf]

# Use FIR approximation for faster speed on GPU.
# This idea is from AudioTools:
# https://github.com/descriptinc/audiotools/blob/master/audiotools/core/loudness.py
# We can set the FIR length with the `zeros` keyword.
meter9 = jln.Meter(rate, use_fir=True, zeros=2048)
```

### Batched operation
Using `jax` allows us to calculate loudness and normalize across a batch dimension using `vmap` (or `pmap` across devices).
Examples from tests:

``` python
def test_batched_integrated_loudness():
    data, rate = sf.read("tests/data/sine_1000.wav")
    meter = pyln.Meter(rate)
    loudness = jax.vmap(meter.integrated_loudness)(jnp.stack([data, data, data]))

    assert jnp.allclose(loudness, jnp.full(loudness.shape, -3.0523438444331137))

def test_batched_loudness_normalize():
    data, rate = sf.read("tests/data/sine_1000.wav")
    data = jnp.stack([data, data, data, data])
    meter = pyln.Meter(rate)
    loudness = jax.vmap(meter.integrated_loudness)(data)
    norm = jax.vmap(pyln.normalize.loudness, in_axes=(0, 0, None))(data, loudness, -6.0)
    loudness = jax.vmap(meter.integrated_loudness)(norm)

    assert jnp.allclose(loudness, jnp.full(loudness.shape, -6.0))
```

`

## Dependencies
- **SciPy** ([https://www.scipy.org/](https://www.scipy.org/))
- **NumPy** ([http://www.numpy.org/](http://www.numpy.org/))
- **JAX** ([https://jax.readthedocs.io/en/latest/index.html](https://jax.readthedocs.io/en/latest/index.html))
- **jaxtyping** ([https://docs.kidger.site/jaxtyping/](https://docs.kidger.site/jaxtyping/))
- **python-soundfile** ([https://github.com/bastibe/python-soundfile](https://github.com/bastibe/python-soundfile))


## References

> Ian Dash, Luis Miranda, and Densil Cabrera, "[Multichannel Loudness Listening Test](http://www.aes.org/e-lib/browse.cfm?elib=14581),"
> 124th International Convention of the Audio Engineering Society, May 2008

> Pedro D. Pestana and Álvaro Barbosa, "[Accuracy of ITU-R BS.1770 Algorithm in Evaluating Multitrack Material](http://www.aes.org/e-lib/online/browse.cfm?elib=16608),"
> 133rd International Convention of the Audio Engineering Society, October 2012

> Pedro D. Pestana, Josh D. Reiss, and Álvaro Barbosa, "[Loudness Measurement of Multitrack Audio Content Using Modifications of ITU-R BS.1770](http://www.aes.org/e-lib/browse.cfm?elib=16714),"
> 134th International Convention of the Audio Engineering Society, May 2013

> Steven Fenton and Hyunkook Lee, "[Alternative Weighting Filters for Multi-Track Program Loudness Measurement](http://www.aes.org/e-lib/browse.cfm?elib=19215),"
> 143rd International Convention of the Audio Engineering Society, October 2017

> Brecht De Man, "[Evaluation of Implementations of the EBU R128 Loudness Measurement](http://www.aes.org/e-lib/browse.cfm?elib=19790)," 
> 145th International Convention of the Audio Engineering Society, October 2018. 

