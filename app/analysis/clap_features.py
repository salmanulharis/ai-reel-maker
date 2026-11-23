"""Audio features for the CLAP sound model (laion/clap-htsat-unfused), in plain numpy.

Matches Hugging Face's ClapFeatureExtractor for this model: 48 kHz mono, 10 s windows
(shorter sound is repeated, then padded with silence), a 1024-point Hann STFT with hop 480,
64 Slaney mel bands from 50 Hz to 14 kHz, and power in decibels. The result has the shape
(1001, 64) per window. A test checks it against the reference implementation.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

SAMPLE_RATE = 48000
WINDOW_SAMPLES = 480000  # 10 s, the length the model was trained on
_N_FFT = 1024
_HOP = 480
_MEL_BANDS = 64
_F_MIN = 50.0
_F_MAX = 14000.0

_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = 15.0
_LOG_STEP = 27.0 / np.log(6.4)


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    freq = np.asarray(freq, dtype=np.float64)
    mels = 3.0 * freq / 200.0
    high = freq >= _MIN_LOG_HZ
    mels[high] = _MIN_LOG_MEL + np.log(freq[high] / _MIN_LOG_HZ) * _LOG_STEP
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    freq = 200.0 * mels / 3.0
    high = mels >= _MIN_LOG_MEL
    freq[high] = _MIN_LOG_HZ * np.exp((mels[high] - _MIN_LOG_MEL) / _LOG_STEP)
    return freq


@lru_cache(maxsize=1)
def mel_filters() -> np.ndarray:
    """(513, 64) Slaney-normalised triangular filters."""
    mel_points = np.linspace(_hz_to_mel(np.array([_F_MIN]))[0], _hz_to_mel(np.array([_F_MAX]))[0], _MEL_BANDS + 2)
    filter_freqs = _mel_to_hz(mel_points)
    fft_freqs = np.linspace(0, SAMPLE_RATE // 2, _N_FFT // 2 + 1)
    diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    filters *= (2.0 / (filter_freqs[2 : _MEL_BANDS + 2] - filter_freqs[:_MEL_BANDS]))[None, :]
    return filters


@lru_cache(maxsize=1)
def _window() -> np.ndarray:
    # Periodic Hann window, as used by the reference extractor.
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_N_FFT) / _N_FFT)


def fit_window(samples: np.ndarray) -> np.ndarray:
    """A 10 s window: longer sound keeps its start, shorter sound is repeated and padded."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size >= WINDOW_SAMPLES:
        return samples[:WINDOW_SAMPLES]
    if samples.size == 0:
        return np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    repeats = WINDOW_SAMPLES // samples.size
    tiled = np.tile(samples, max(1, repeats))
    return np.pad(tiled, (0, WINDOW_SAMPLES - tiled.size))


def log_mel(samples: np.ndarray) -> np.ndarray:
    """(1001, 64) log-mel features of one window of 48 kHz samples in -1..1."""
    wave = fit_window(samples).astype(np.float64)
    padded = np.pad(wave, (_N_FFT // 2, _N_FFT // 2), mode="reflect")
    frames = 1 + (padded.size - _N_FFT) // _HOP
    index = np.arange(_N_FFT)[None, :] + _HOP * np.arange(frames)[:, None]
    spectrum = np.abs(np.fft.rfft(padded[index] * _window(), n=_N_FFT)) ** 2
    mel = spectrum @ mel_filters()
    return (10.0 * np.log10(np.maximum(mel, 1e-10))).astype(np.float32)
