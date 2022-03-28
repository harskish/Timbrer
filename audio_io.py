import numpy as np
from pathlib import Path
from pydub import AudioSegment
from constants import sample_rate as DEFAULT_SAMPLE_RATE

# Based on: https://github.com/rec/vl8/blob/main/old/vl8/dsp/pydub_io.py

def read(filename, offset=None, duration=None, mono=True):
    s = AudioSegment.from_file(filename, start_second=offset, duration=duration).set_frame_rate(DEFAULT_SAMPLE_RATE)
    if mono:
        s = s.set_channels(1)
    array = np.frombuffer(s._data, dtype=s.array_type)
    nsamples = len(array) // s.channels

    # To float32
    array = array.astype(np.float32) / 2**(8*s.frame_width-1)

    assert not len(array) % s.channels
    assert nsamples == int(s.frame_count())

    matrix = array.reshape((s.channels, nsamples), order='F')
    return np.squeeze(matrix), s.frame_rate

def write(data, filename, sample_rate=DEFAULT_SAMPLE_RATE, *args, **kwargs):
    first, *rest = data.shape

    if not np.issubdtype(data.dtype, np.integer):
        m1, m2 = np.amax(data), -np.amin(data)
        m = max(m1, m2)
        if m > 1:
            data = data / m

        data = 0xFFFF * (1 + data) / 2 - 0x8000
        assert np.amax(data) <= 0x7FFF
        assert np.amin(data) >= -0x8000
        data = data.astype(np.int16)

    segment = AudioSegment(
        data=data.tobytes('F'),
        sample_width=data.dtype.itemsize,
        frame_rate=sample_rate,
        channels=first if rest else 1,
    )

    format = Path(filename).suffix[1:]
    assert format, 'Invalid output format "{}"'
    
    return segment.export(filename, format, *args, **kwargs)