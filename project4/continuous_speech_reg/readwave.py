import pyaudio
import wave
import struct
import record


def read_wave(filename=record.WAVE_OUTPUT_FILENAME):
    wave_samples=[]
    wf=wave.open(filename,'rb')
    data =wf.readframes(record.CHUNK)
    while data != b'':
        wave_samples.extend(struct.unpack("%dh" %  (len(data)// 2), data))
        data = wf.readframes(record.CHUNK)
    return  wave_samples