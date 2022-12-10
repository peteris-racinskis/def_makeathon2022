import numpy as np
from os import listdir

SAMPLE_DIR="audio_samples"
sample_files = list(filter(lambda s: "bin" in s, listdir(SAMPLE_DIR)))

class AudioSample():

    def __init__(self, filename):
        self.data = read_sample(filename)
        self.interval = t_diff(filename)

def t_diff(f):
    start, end = [int(x) for x in f.split("/")[-1].split(".")[0].split("-")]
    return float(end - start) / 1e9

def read_sample(filename) -> np.ndarray:
    with open(filename, 'rb') as f:
        buf = f.read()
    sample = np.frombuffer(buf, dtype=np.int16).reshape(-1,4)
    return sample

samples = []
for sf in sample_files:
    samples.append(AudioSample(f"{SAMPLE_DIR}/{sf}"))

print()
