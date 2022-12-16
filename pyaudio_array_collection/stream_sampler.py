import numpy as np
import pyaudio
import matplotlib.pyplot as plt

from time import sleep
from math import floor, ceil

from pose_subscriber import PoseSubscriber
from utils import get_usb_sound_cards, add_buffer, visual_normalization as vnorm

CHANNELS=1
FORMAT=pyaudio.paInt16
RATE=44100
CHUNK=4096
RECORD_SECS = 10
TOTAL_LEN = ceil( RATE / CHUNK ) * CHUNK * RECORD_SECS * 2 # for uint16 we have 2 bytes per value
SLICES_PER_SECOND = 100
TOTAL_SLICES = int( SLICES_PER_SECOND * RECORD_SECS )
SLICE_OFFSET = int( RATE / SLICES_PER_SECOND )
DISTANCE_THRESH = 0.5

class StreamSampler():

    def __init__(self, sub=PoseSubscriber()):
        self.sub = sub
        self.p = pyaudio.PyAudio()
        self.idxs = get_usb_sound_cards(self.p)
        assert(len(self.idxs) == 3)
        self.slice_rate = 10
        self.samples = []
    
    def sample_once(self):
        
        streams, buffers = list(zip(*[self.create_stream(idx) for idx in self.idxs]))
        poses = []

        for stream in streams:
            stream.start_stream()
        
        while any([stream.is_active() for stream in streams]):
            poses.append(self.sub.get_pos())
            sleep( 1 / SLICES_PER_SECOND )
        
        pose_arr = np.stack(poses)
        moved = np.linalg.norm(pose_arr[0] - pose_arr[-1])
        pos_samples = pose_arr[ floor( SLICES_PER_SECOND / 2 ) : TOTAL_SLICES - ceil( SLICES_PER_SECOND / 2 ) ]

        valid = moved < DISTANCE_THRESH        
        if valid:
            self.samples.append( 
                (
                    np.stack(self.process_sample(b) for b in buffers), 
                    np.stack(pos_samples)
                ) 
            )
            print(f"Valid sample collected, drone moved {moved}, total {len(self.samples)}")
        else:
            print(f"Maximum distance for window exceeded, drone moved {moved}")
        
        for stream in streams:
            stream.stop_stream()
            stream.close()

    @staticmethod
    def process_sample(buf):

        ar = np.frombuffer(buf, dtype=np.int16).reshape(-1, int( TOTAL_LEN / 2 ) )
        list_slice_arrays = []

        for i in range(TOTAL_SLICES - SLICES_PER_SECOND - 1):
            start = i * SLICE_OFFSET
            stop = start + RATE
            list_slice_arrays.append(ar[:,start:stop])
        
        slice_array = np.stack(list_slice_arrays)
        return np.abs(np.fft.rfft(slice_array, axis=-1))
    

    def create_stream(self, index):
        buffer = bytearray()
        self.p.is_format_supported(RATE,index,CHANNELS,FORMAT) ## exception if fail
        stream = self.p.open(
            input=True,
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            frames_per_buffer=CHUNK,
            input_device_index=index,
            stream_callback=add_buffer(self.collect_sample_callback, buffer),
            start=False
        )
        return stream, buffer
    
    @staticmethod
    def collect_sample_callback(in_data, frame_count, time_info, status_flags, buffer=None):
        buffer += in_data
        cont = pyaudio.paContinue if len(buffer) < TOTAL_LEN else pyaudio.paComplete
        return (None, cont)    

    def visualize_sample(self, idx):
        samples, positions = self.samples[idx]
        slen = len(positions)
        fig, axs = plt.subplots(3,2)
        axs[0][0].matshow(vnorm(samples[0])[:,:2000])
        axs[1][0].matshow(vnorm(samples[1])[:,:2000])
        axs[2][0].matshow(vnorm(samples[2])[:,:2000])

        axs[0][1].plot(np.arange(slen), positions[:,0])
        axs[1][1].plot(np.arange(slen), positions[:,1])
        axs[2][1].plot(np.arange(slen), positions[:,2])
        
        fig.show()



if __name__ == "__main__":
    ss = StreamSampler()
    ss.sample_once()
    # ss.sample_once()
    # ss.sample_once()
    # ss.sample_once()
    ss.visualize_sample(0)
    print()