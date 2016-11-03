import math
import numpy as np

# parameters for wave_sample
SAMPLE_NUMBER = -99
SAMPLE_RATE = 16000
SAMPLE_SIZE = 400
MIN_FREQ = 0  # 133.33
MAX_FREQ = 8000  # 6855.5
# parameters for preemph
PREAMPH_A = 0.95
# parameters for windowing
WINDOW_FUNCTION=[(0.54 - 0.46 * math.cos(2 * math.pi * i / (SAMPLE_SIZE - 1))) for i in range(SAMPLE_SIZE)]
# paraeters for zero_padding
PADDING_SIZE = 512
# parameters for logmel
FILTER_NUM = 40
# parameters for dct
DCT_DIM = 13
# global parameters
# Debug control
DEBUG = 1


def preemph(recordsamples):
    record_sample_len=len(recordsamples)
    data_preemph = [0 for i in range(record_sample_len)]
    data_preemph[0] = recordsamples[0]
    for i in range(1, record_sample_len):
        data_preemph[i] = recordsamples[i] - (recordsamples[i - 1] * PREAMPH_A)
    return data_preemph


def convert_to_frame(data_preemph, start_index):
    frame_data = [0 for i in range(SAMPLE_SIZE)]
    if (start_index + SAMPLE_SIZE - 1) < SAMPLE_NUMBER:
        for i in range(0, SAMPLE_SIZE):
            frame_data[i] = data_preemph[start_index + i]
    else:
        for i in range(0, SAMPLE_NUMBER - start_index):
            frame_data[i] = data_preemph[start_index + i]
    return frame_data


def window(frame):
    windowed_frame = []
    frame_len = len(frame)
    for i in range(frame_len):
        windowed_frame.append(frame[i] * WINDOW_FUNCTION[i])
    return windowed_frame


def zero_padding(frame):
    for i in range(SAMPLE_SIZE, PADDING_SIZE):
        frame.append(0)
    return frame


def hz2mel(hz):
    return 2595 * math.log10(1 + hz / 700)


def mel2hz(mel):
    return 700 * (math.pow(10, mel / 2595) - 1)


def generate_mi():
    """"generate MI matrix for logmel procedure"""
    fftfrqs = []
    for i in range(PADDING_SIZE // 2 + 1):
        fre = i * SAMPLE_RATE / PADDING_SIZE
        fftfrqs.append(fre)
    minmel = hz2mel(MIN_FREQ)
    maxmel = hz2mel(MAX_FREQ)
    binfrqs = []
    for i in range(0, FILTER_NUM + 2):
        binfrqs.append(mel2hz(minmel + i * (maxmel - minmel) / (FILTER_NUM + 1)))
    MI = [[0 for i in range(PADDING_SIZE // 2 + 1)] for i in range(FILTER_NUM)]
    for i in range(FILTER_NUM):
        low = binfrqs[i]
        mid = binfrqs[i + 1]
        high = binfrqs[i + 2]
        for j in range(int(PADDING_SIZE / 2 + 1)):
            low_slope = (fftfrqs[j] - low) / (mid - low)
            high_slope = (high - fftfrqs[j]) / (high - mid)
            min_value=low_slope
            if low_slope > high_slope:
                min_value =high_slope
            if min_value > 0:
                MI[i][j] = min_value
            else:
                MI[i][j] = 0
    return MI


def dct(log_mel_spectrum):
    mel_ceptrum = []
    for i in range(DCT_DIM):
        sum = 0
        for j in range(FILTER_NUM):
            sum += log_mel_spectrum[j] * math.cos((j + 0.5) * math.pi / FILTER_NUM * i)
        if i == 0:
            mel_ceptrum.append(sum * math.sqrt(1.0 / FILTER_NUM))
        else:
            mel_ceptrum.append(sum * math.sqrt(2.0 / FILTER_NUM))
    return mel_ceptrum


def mfcc(wave,mode="normal"):
    """generate 39 dimensions dct matrix-- the mel_ceptrum_matrix from the input wave sample list"""
    #prepare
    mel_ceptrum_matrix = []
    logmel_matrix = []
    #openfile for output data
    if mode=="normal":
        file_wave=open('wave.txt','w')
        file_preemphwave=open('preemph_wav.txt','w')
        file_window = open('window.txt', 'w')
        file_zeropad = open('zeroPad.txt', 'w')
        file_powerspectrum = open('powerspectrum.txt', 'w')
        file_logmel = open('logmel.txt', 'w')
        file_dct13 = open('dct13.txt', 'w')
        file_ndct13 = open('ndct13.txt', 'w')
        file_ndct39 = open('ndct39.txt', 'w')
    MI = generate_mi()

    # read source wave
    global SAMPLE_NUMBER
    SAMPLE_NUMBER= len(wave)

    if mode=="normal":
        for value in wave:
            file_wave.write(str(value)+ ' ')

    # pre-emphasize
    wave = preemph(wave)

    if mode=="normal":
        for value in wave:
            file_preemphwave.write(str(value)+ ' ')

    # cut frame and process each frame
    sample_index = 0
    sample_boundary = SAMPLE_NUMBER-400
    while sample_index <= sample_boundary:
        # frame prepare
        frame = []
        frame = convert_to_frame(wave, sample_index)

        # window
        frame = window(frame)

        if mode == "normal":
            for value in frame:
                file_window.write(str(value) + ' ')
                file_window.write('\n')

        # zeroPad
        frame = zero_padding(frame)

        if mode == "normal":
            for value in frame:
                file_zeropad.write(str(value) + ' ')
            file_zeropad.write('\n')

        # fft and cal power spectrum
        power_spectrum = []
        fft_result = np.fft.fft(frame)
        for value in fft_result:
            power_spectrum.append(value.real ** 2 + value.imag ** 2)
        power_spectrum = power_spectrum[:len(power_spectrum) // 2 + 1]

        if mode == "normal":
            for value in power_spectrum:
                file_powerspectrum.write(str(value) + ' ')
            file_powerspectrum.write('\n')

        # cal logmel
        logmel_spectrum = []
        for i in range(FILTER_NUM):
            sum = 0
            for j in range(PADDING_SIZE // 2 + 1):
                sum += power_spectrum[j] * MI[i][j]
            logmel_spectrum.append(math.log(sum))
        logmel_matrix.append(logmel_spectrum)

        if mode == "normal":
            for value in logmel_spectrum:
                file_logmel.write(str(value) + ' ')
            file_logmel.write('\n')

        # dct
        mel_ceptrum = dct(logmel_spectrum)
        if mode == "normal":
            for value in mel_ceptrum:
                file_dct13.write(str(value) + ' ')
            file_dct13.write('\n')
        # store mel_ceptrum
        mel_ceptrum_matrix.append(mel_ceptrum)
        # while control
        sample_index += 160

    # normalization
    # cal mean
    mean_mel_ceptrum=[0]*13
    mel_ceptrum_matrix_len = len(mel_ceptrum_matrix)
    mel_ceptrum_len = len(mel_ceptrum_matrix[0])
    for mel_ceptrum in mel_ceptrum_matrix:
        for i in range(mel_ceptrum_len):
            mean_mel_ceptrum[i]+=mel_ceptrum[i]
    mean_mel_ceptrum= list(map(lambda x: x/mel_ceptrum_matrix_len, mean_mel_ceptrum))

    # minus min
    for mel_ceptrum_index in range(mel_ceptrum_matrix_len):
        for mel_ceptrum_inner_index in range(mel_ceptrum_len):
            mel_ceptrum_matrix[mel_ceptrum_index][mel_ceptrum_inner_index] -= mean_mel_ceptrum[mel_ceptrum_inner_index]

    # cal standard deviation
    sd=[0]*13
    for mel_ceptrum in mel_ceptrum_matrix:
        for i in range(mel_ceptrum_len):
            sd[i]+=mel_ceptrum[i]**2

    sd=list(map(lambda x: (x/mel_ceptrum_matrix_len)**0.5,sd))
    # devide sd
    for mel_ceptrum_index in range(mel_ceptrum_matrix_len):
        for mel_ceptrum_inner_index in range(mel_ceptrum_len):
            mel_ceptrum_matrix[mel_ceptrum_index][mel_ceptrum_inner_index] /= sd[mel_ceptrum_inner_index]

    # normalization done

    # output file
    if mode == "normal":
        for single_mel_ceptrum in mel_ceptrum_matrix:
            for value in single_mel_ceptrum:
                file_ndct13.write(str(value) + ' ')
            file_ndct13.write('\n')

    # extend ceptrum
    for i in range(mel_ceptrum_matrix_len-1):
        mel_ceptrum_matrix[i].extend(list(map(lambda x:x[0]-x[1],zip(mel_ceptrum_matrix[i+1],mel_ceptrum_matrix[i]))))
    mel_ceptrum_matrix[mel_ceptrum_matrix_len-1].extend(mel_ceptrum_matrix[mel_ceptrum_matrix_len-1])

    for i in range(mel_ceptrum_matrix_len-1):
        mel_ceptrum_matrix[i].extend(list(map(lambda x: x[0] - x[1], zip(mel_ceptrum_matrix[i + 1][13:], mel_ceptrum_matrix[i][13:]))))
    mel_ceptrum_matrix[mel_ceptrum_matrix_len-1].extend(mel_ceptrum_matrix[mel_ceptrum_matrix_len-1][13:])

    if mode == "normal":
        for single_mel_ceptrum in mel_ceptrum_matrix:
            for value in single_mel_ceptrum:
                file_ndct39.write(str(value) + ' ')
            file_ndct39.write('\n')

    #file close
    if mode == "normal":
        file_preemphwave.close()
        file_window.close()
        file_zeropad.close()
        file_powerspectrum.close()
        file_logmel.close()
        file_dct13.close()
        file_ndct13.close()
        file_ndct39.close()
    return mel_ceptrum_matrix

# if __name__ ==  '__main__':
#     cProfile.run("mfcc()")

