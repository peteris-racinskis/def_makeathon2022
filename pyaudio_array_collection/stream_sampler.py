import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from time import sleep, time
from math import floor, ceil

from pose_subscriber import PoseSubscriber
from utils import get_usb_sound_cards, add_buffer, visual_normalization as vnorm

CHANNELS=1
FORMAT=pyaudio.paInt16
RATE=44100
WINDOWS_PER_SECOND=4 # width of the sample window for each slice (how many samples get FFT'd)
CHUNK=4096
RECORD_SECS = 1
TOTAL_LEN = ceil( RATE / CHUNK ) * CHUNK * RECORD_SECS * 2 # for uint16 we have 2 bytes per value
SLICES_PER_SECOND = 180
TOTAL_SLICES = int( SLICES_PER_SECOND * RECORD_SECS )
SLICE_OFFSET = int( RATE / SLICES_PER_SECOND )
DISTANCE_THRESH_MAX = 1.0
DISTANCE_THRESH_MIN = 1e-2
RECORD_MAX_TIME_MULTIPLIER = 2.0

class StreamSampler():

    def __init__(self, sub=PoseSubscriber()):
        self.sub = sub
        self.p = pyaudio.PyAudio()
        self.idxs = get_usb_sound_cards(self.p)
        # assert(len(self.idxs) == 3)
        self.slice_rate = 10
        self.samples = []
    
    def collect_and_save(self, num_samples=100, filename=f"stream_samples/<item>_{time()}.npy"):

        t_start = time()
        dt_max = RECORD_MAX_TIME_MULTIPLIER * num_samples * RECORD_SECS # if drone dies we land it and don't reecord any more

        while len(self.samples) < num_samples:
            self.sample_once()
            if time() - t_start > dt_max:
                break

        sname = filename.replace("<item>", "samples")
        pname = filename.replace("<item>", "positions")
        print(f"Saving samples to file: {sname}")
        print(f"Saving positions to file: {pname}")

        combined_audio = np.stack([sample for sample,_ in self.samples])
        combined_pos = np.stack([pos for _,pos in self.samples])

        np.save(sname, combined_audio)
        np.save(pname, combined_pos)
        
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
        pos_samples = poses[ floor( SLICES_PER_SECOND / ( 2 * WINDOWS_PER_SECOND ) ) :
         TOTAL_SLICES - ceil( SLICES_PER_SECOND / ( 2 * WINDOWS_PER_SECOND ) ) ]

        valid = DISTANCE_THRESH_MIN < moved < DISTANCE_THRESH_MAX 
        if valid:
            self.samples.append( 
                (
                    np.stack(self.process_sample(b) for b in buffers), 
                    np.stack(pos_samples)
                ) 
            )
            print(f"Valid sample collected, drone moved {moved}, total {len(self.samples)}")
        elif moved >= DISTANCE_THRESH_MAX:
            print(f"Maximum distance for window exceeded, drone moved {moved}")
        elif moved <= DISTANCE_THRESH_MIN:
            print(f"Minimum distance for window not attained, drone moved {moved}")
        
        for stream in streams:
            stream.stop_stream()
            stream.close()

    @staticmethod
    def process_sample(buf):

        ar = np.frombuffer(buf, dtype=np.int16).reshape( int( TOTAL_LEN / 2 ) )
        list_slice_arrays = []

        for i in range( TOTAL_SLICES - ceil( SLICES_PER_SECOND / WINDOWS_PER_SECOND ) ):
            start = i * SLICE_OFFSET
            stop = start + floor( RATE / WINDOWS_PER_SECOND ) 
            list_slice_arrays.append(ar[start:stop])
        
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

    @staticmethod
    def visualize_sample_and_pos(samples, positions, fig=None, axs=None):

        slen = len(positions)
        must_close = False
        if fig is None:
            must_close = True
            fig, axs = plt.subplots(3,2)

        axs[0][0].clear()
        axs[1][0].clear()
        axs[2][0].clear()
        axs[1][1].clear()
        axs[2][1].clear()

        axs[0][0].matshow(vnorm(samples[0]), aspect="auto")
        axs[1][0].matshow(vnorm(samples[1]), aspect="auto")
        axs[2][0].matshow(vnorm(samples[2]), aspect="auto")

        path = Line2D(positions[:,0], positions[:,1], color="blue")
        axs[0][1].scatter([0.2], [0], color="red", s=8)
        axs[0][1].scatter([0], [0.2], color="green", s=8)
        axs[0][1].scatter([0], [0], color="blue", s=8)
        axs[0][1].add_line(path)
        # axs[0][1].scatter(positions[:,0], positions[:,1])
        axs[0][1].set_xlim(2.5,-2.5)
        axs[0][1].set_ylim(2.5,-2.5)
        axs[0][1].set_aspect("equal")
        # axs[0][1].plot(np.arange(slen), positions[:,0])
        axs[1][1].plot(np.arange(slen), positions[:,0], color="red")
        axs[1][1].plot(np.arange(slen), positions[:,1], color="green")
        axs[2][1].plot(np.arange(slen), positions[:,2])

        fig.show()
        plt.pause(1)
        if must_close:
            plt.close(fig)
        

    def visualize_sample(self, idx):

        samples, positions = self.samples[idx]
        self.visualize_sample_and_pos(samples, positions)


if __name__ == "__main__":
    ss = StreamSampler()
    # ss.sample_once()
    # ss.collect_and_save(600)
    positions = np.load("stream_samples/positions_1671276025.1548362.npy")
    samples = np.load("stream_samples/samples_1671276025.1548362.npy")
    # print()
    fig, axs = plt.subplots(3,2)
    for s, p in zip(samples, positions):
        ss.visualize_sample_and_pos(s, p, fig, axs)
    # ss.visualize_sample(0)
    print()