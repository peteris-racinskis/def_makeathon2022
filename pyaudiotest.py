import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from time import sleep

CHUNK=1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECS = 2
BUFFERS = int( ( RATE / CHUNK ) * RECORD_SECS )
TOTAL_LEN = int( RATE / CHUNK ) * CHUNK * RECORD_SECS * 2 # for uint16 we have 2 bytes per value


def add_buffer(cb, buffer):
    return partial(cb, buffer=buffer)

def collect_sample_callback(in_data, frame_count, time_info, status_flags, buffer=None):
    buffer += in_data
    cont = pyaudio.paContinue if len(buffer) < TOTAL_LEN else pyaudio.paComplete
    return (None, cont)    

def get_usb_sound_cards(p: pyaudio.PyAudio):
    usb_sound_card_indices = []
    for i in range(p.get_device_count()):
        devinfo = p.get_device_info_by_index(i)
        if 'USB PnP Sound Device: Audio' in devinfo['name']:
            usb_sound_card_indices.append(devinfo['index'])
    return usb_sound_card_indices

def create_stream(index, p):
    buffer = bytearray()
    p.is_format_supported(RATE,index,CHANNELS,FORMAT) ## exception if fail
    stream = p.open(
        input=True,
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        frames_per_buffer=CHUNK,
        input_device_index=index,
        stream_callback=add_buffer(collect_sample_callback, buffer),
        start=False
    )
    return stream, buffer

def collect_3_streams():

    p = pyaudio.PyAudio()

    usb1, usb2, usb3 = get_usb_sound_cards(p)

    s1, b1 = create_stream(usb1, p)
    s2, b2 = create_stream(usb2, p)
    s3, b3 = create_stream(usb3, p)

    s1.start_stream()
    s2.start_stream()
    s3.start_stream()

    while s3.is_active():
        sleep(0.01)
        continue

    s1.stop_stream()
    s2.stop_stream()
    s3.stop_stream()

    return b1, b2, b3

def array_from_buffer(buf):
    return np.frombuffer(buf, dtype=np.int16).reshape(-1, TOTAL_LEN)


if __name__ == "__main__":

    b1, b2, b3 = collect_3_streams()
    print()



# buffers_a = []
# buffers_b = []
# for _ in range(0, BUFFERS):
#     data = stream_a.read(CHUNK, exception_on_overflow=False)
#     buffers_a.append(data)
#     data = stream_b.read(CHUNK, exception_on_overflow=False)
#     buffers_b.append(data)

# arrays_a = []
# arrays_b = []
# for buf in buffers_a:
#     ar = np.frombuffer(buf, dtype = np.int16)
#     arrays_a.append(ar)
# for buf in buffers_b:
#     ar = np.frombuffer(buf, dtype = np.int16)
#     arrays_b.append(ar)

# time_time_plot_a = np.array(arrays_a).reshape(-1, CHUNK)
# time_freq_plot_a = np.abs(np.fft.rfft(time_time_plot_a, axis=-1))
# time_time_plot_b = np.array(arrays_b).reshape(-1, CHUNK)
# time_freq_plot_b = np.abs(np.fft.rfft(time_time_plot_b, axis=-1))

# plt.matshow(np.log(20 * time_freq_plot_a[:,10:].T))
# plt.matshow(np.log(20 * time_freq_plot_b[:,10:].T))
# plt.show()

# print()