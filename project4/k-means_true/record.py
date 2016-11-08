import pyaudio
import wave
import math
import os
import struct

# parameters for recording
CHUNK = 400
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 60.0
WAVE_OUTPUT_FILENAME = "output.wav"
# parameters in classify_frame function
FORGET_FACTOR = 1.0
ADJUSTMENT = 0.05
THRESHOLD = 2.0
level = 0.0
background = 0.0
# parameters in end pointing
continue_silence = 0.0
MIN_SILENCE_LEN = 0.2
frame_number = 0


def cal_frame_energy_in_decibel(frame):
    energy = 0
    for sample in frame:
        energy+=sample**2
    return 10 * math.log10(energy)


def classify_frame(frame):
    global level
    global background
    global continue_silence
    current=cal_frame_energy_in_decibel(frame)
    is_speech=False
    level = ((level * FORGET_FACTOR) + current) / (FORGET_FACTOR + 1)
    if current < background:
        background=level
    else:
        background += (current-background)*ADJUSTMENT
    if level<background:
        level=background
    if (level - background) > THRESHOLD:
        is_speech=True
        continue_silence=0
    else:
        continue_silence += 1.0*len(frame)/RATE
        if continue_silence< MIN_SILENCE_LEN: # if it is still less than min_silence_len, we assume it is still a speech
            is_speech=True
    return is_speech


# start recording
def record():
    """record a wave from mic, save it as output.wav file and return the wave samples"""
    global frame_number
    global background
    global level
    wave_samples=[]
    frames=[]
    p = pyaudio.PyAudio()
    os.system("pause")
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
    print("* recording")

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        ss = struct.unpack("%dh" % (len(data) // 2), data)
        wave_samples.extend(ss)
        if frame_number == 0:
            level = cal_frame_energy_in_decibel(ss)
        if frame_number < 10:
            background += cal_frame_energy_in_decibel(ss)
        if frame_number == 9:
            background /= 10.0
        if frame_number >= 10:
            if classify_frame(ss):
                pass
            else:
                break
        frame_number += 1
        frames.append(data)
        print("recording level-background=", level, cal_frame_energy_in_decibel(ss), background, level-background, continue_silence)
    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return wave_samples

