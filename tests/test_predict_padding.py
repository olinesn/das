"""`predict` must annotate a recording up to its last sample, whatever its length and the batch size."""

from das import predict  # first: das selects Keras's torch backend

import keras
import numpy as np
import pytest

NB_HIST = 64
DATA_PADDING = 8
FS = 1_000


def _always_song_model():
    """Predicts "song" (class 1) with probability ~1 for every sample."""
    model = keras.Sequential([keras.Input(shape=(NB_HIST, 1)), keras.layers.Dense(2), keras.layers.Softmax()])
    model.set_weights([np.zeros((1, 2), np.float32), np.array([-10.0, 10.0], np.float32)])
    return model


def _params(batch_size):
    return {
        "samplerate_x_Hz": FS,
        "nb_hist": NB_HIST,
        "stride": NB_HIST - 2 * DATA_PADDING,
        "data_padding": DATA_PADDING,
        "batch_size": batch_size,
        "nb_classes": 2,
        "class_names": ["noise", "song"],
        "class_types": ["segment", "segment"],
    }


@pytest.mark.parametrize("batch_size", [1, 4, 7])
@pytest.mark.parametrize("batches,extra", [(10, 0), (10, -1), (10, -5), (10, 5), (10, -300), (1, 3), (0, 100)])
def test_predict_reaches_the_end_of_the_recording(batch_size, batches, extra):
    batch_len = batch_size * NB_HIST + NB_HIST  # predict() pads x to a multiple of this
    n = batches * batch_len + extra
    if n <= NB_HIST:
        pytest.skip("shorter than one chunk")
    x = np.zeros((n, 1), np.float32)

    _, segments, probs, _ = predict.predict(
        x, model=_always_song_model(), params=_params(batch_size), verbose=0, segment_minlen=None, segment_fillgap=None
    )

    assert probs.shape == (n, 2)
    # the first DATA_PADDING samples are never predicted (the network ignores the edges of each chunk)
    assert np.all(probs[DATA_PADDING:, 1] > 0.5)
    np.testing.assert_allclose(segments["onsets_seconds"], [DATA_PADDING / FS])
    np.testing.assert_allclose(segments["offsets_seconds"], [(n - 1) / FS])


def test_batch_size_does_not_change_predictions():
    rng = np.random.default_rng(0)
    model = keras.Sequential([keras.Input(shape=(NB_HIST, 1)), keras.layers.Dense(2), keras.layers.Softmax()])
    x = rng.normal(size=(3_333, 1)).astype(np.float32)
    ref = predict.predict(x, model=model, params=_params(4), verbose=0)[2]
    for batch_size in (1, 3, 16):
        probs = predict.predict(x, model=model, params=_params(batch_size), verbose=0)[2]
        np.testing.assert_allclose(probs, ref, atol=1e-6)
