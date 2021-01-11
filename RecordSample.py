import os
import wave
import time
import pickle
import pyaudio
import random
import string


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 10
device_index = 2
audio = pyaudio.PyAudio()
print("----------------------record device list---------------------")
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
    if (audio.get_device_info_by_host_api_device_index(0, i)
            .get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ",
              audio.get_device_info_by_host_api_device_index(0, i).get('name'))
print("-------------------------------------------------------------")
index = int(input())
for i in range(0, 47):
    audio = pyaudio.PyAudio()
    print("recording via index "+str(index))
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=index,
                        frames_per_buffer=CHUNK)
    print("recording started")
    Recordframes = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    print("recording stopped")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    # OUTPUT_FILENAME = get_random_string(8)+".wav"
    OUTPUT_FILENAME = "Grzegorz375.wav"
    WAVE_OUTPUT_FILENAME = os.path.join(
        "testing_samples", OUTPUT_FILENAME)
    print(WAVE_OUTPUT_FILENAME)
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()
