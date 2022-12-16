import pyaudio
import numpy as np

from functools import partial
from time import sleep

CHUNK=1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECS = 2
BUFFERS = int( ( RATE / CHUNK ) * RECORD_SECS )
TOTAL_LEN = int( RATE / CHUNK ) * CHUNK * RECORD_SECS * 2 # for uint16 we have 2 bytes per value

def visual_normalization(fft_result):
    return np.log(20 * fft_result.squeeze()[:,1:])

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