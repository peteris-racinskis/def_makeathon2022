import pyaudio
import numpy as np

from functools import partial
from time import sleep

# CHUNK=1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 44100
# RECORD_SECS = 2
# BUFFERS = int( ( RATE / CHUNK ) * RECORD_SECS )
# TOTAL_LEN = int( RATE / CHUNK ) * CHUNK * RECORD_SECS * 2 # for uint16 we have 2 bytes per value

def visual_normalization(fft_result):
    return np.log(20 * fft_result.squeeze()[:,10:])
    # return np.log(20 * fft_result.squeeze()[:,10:])

def add_buffer(cb, buffer):
    return partial(cb, buffer=buffer)


def get_usb_sound_cards(p: pyaudio.PyAudio):
    usb_sound_card_indices = []
    for i in range(p.get_device_count()):
        devinfo = p.get_device_info_by_index(i)
        if 'USB PnP Sound Device: Audio' in devinfo['name']:
            usb_sound_card_indices.append(devinfo['index'])
    return usb_sound_card_indices
