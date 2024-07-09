from pathlib import Path

import jax
import jaxloudnorm as pyln
import soundfile as sf
import jax.numpy as jnp
import pytest

data_dir = Path(__file__).parent / "data"


@pytest.mark.parametrize("use_fir", [False, True])
def test_integrated_loudness(use_fir: bool):
    data, rate = sf.read(data_dir / "sine_1000.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=4096)
    loudness = meter.integrated_loudness(data)

    assert jnp.isclose(loudness, -3.0523438444331137)


@pytest.mark.parametrize("use_fir", [False, True])
def test_batched_integrated_loudness(use_fir: bool):
    data, rate = sf.read(data_dir / "sine_1000.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=4096)
    loudness = jax.vmap(meter.integrated_loudness)(jnp.stack([data, data, data]))

    assert jnp.allclose(loudness, jnp.full(loudness.shape, -3.0523438444331137))


@pytest.mark.parametrize("use_fir", [False, True])
def test_peak_normalize(use_fir: bool):
    data = jnp.array(0.5)
    norm = pyln.normalize.peak(data, 0.0)

    assert jnp.isclose(norm, 1.0)


@pytest.mark.parametrize("use_fir", [False, True])
def test_loudness_normalize(use_fir: bool):
    data, rate = sf.read(data_dir / "sine_1000.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)
    norm = pyln.normalize.loudness(data, loudness, -6.0)
    loudness = meter.integrated_loudness(norm)

    assert jnp.isclose(loudness, -6.0)


@pytest.mark.parametrize("use_fir", [False, True])
def test_batched_loudness_normalize(use_fir: bool):
    data, rate = sf.read(data_dir / "sine_1000.wav")
    data = jnp.stack([data, data, data, data])
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = jax.vmap(meter.integrated_loudness)(data)
    norm = jax.vmap(pyln.normalize.loudness, in_axes=(0, 0, None))(data, loudness, -6.0)
    loudness = jax.vmap(meter.integrated_loudness)(norm)

    assert jnp.allclose(loudness, jnp.full(loudness.shape, -6.0))


@pytest.mark.parametrize("use_fir", [False, True])
def test_rel_gate_test(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Comp_RelGateTest.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -10.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def test_abs_gate_test(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Comp_AbsGateTest.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -69.5
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def test_24LKFS_25Hz_2ch(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Comp_24LKFS_25Hz_2ch.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=2048)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def test_24LKFS_100Hz_2ch(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Comp_24LKFS_100Hz_2ch.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def test_24LKFS_500Hz_2ch(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Comp_24LKFS_500Hz_2ch.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def test_24LKFS_1000Hz_2ch(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Comp_24LKFS_1000Hz_2ch.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def test_24LKFS_2000Hz_2ch(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Comp_24LKFS_2000Hz_2ch.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def test_24LKFS_10000Hz_2ch(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Comp_24LKFS_10000Hz_2ch.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def test_23LKFS_25Hz_2ch(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Comp_23LKFS_25Hz_2ch.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=2048)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def test_23LKFS_100Hz_2ch(use_fir: bool):
    data, rate = sf.read("data/1770-2_Comp_23LKFS_100Hz_2ch.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def test_23LKFS_500Hz_2ch(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Comp_23LKFS_500Hz_2ch.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def test_23LKFS_1000Hz_2ch(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Comp_23LKFS_1000Hz_2ch.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def test_23LKFS_2000Hz_2ch(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Comp_23LKFS_2000Hz_2ch.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def test_23LKFS_10000Hz_2ch(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Comp_23LKFS_10000Hz_2ch.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def test_18LKFS_frequency_sweep(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Comp_18LKFS_FrequencySweep.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -18.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def test_conf_stereo_vinL_R_23LKFS(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Conf_Stereo_VinL+R-23LKFS.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def test_conf_monovoice_music_24LKFS(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Conf_Mono_Voice+Music-24LKFS.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def conf_monovoice_music_24LKFS(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Conf_Mono_Voice+Music-24LKFS.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1


@pytest.mark.parametrize("use_fir", [False, True])
def test_conf_monovoice_music_23LKFS(use_fir: bool):
    data, rate = sf.read(data_dir / "1770-2_Conf_Mono_Voice+Music-23LKFS.wav")
    meter = pyln.Meter(rate, use_fir=use_fir, zeros=1024)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert targetLoudness - 0.1 <= loudness <= targetLoudness + 0.1
