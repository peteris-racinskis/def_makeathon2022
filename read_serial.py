from serial import Serial
from time import time_ns, time, sleep
import numpy as np
from os.path import exists


OUTPUT_FILE = f"audio_samples/freqdomain_{time()}.csv"
S = Serial("/dev/ttyUSB0", baudrate=115200, timeout=3)

class Sample():

    def __init__(self, buf, ts):
        self.timestamp = ts
        self.data, self.valid = parse_sample(buf)
        self.data = self.data.astype(float) / 256
        self.data = np.absolute(np.fft.rfft(self.data, axis=0))
    
    def serialize(self):
        return f"{self.timestamp}," + ",".join(str(x) for x in self.data.reshape(-1, order="F")) + "\n"

def parse_sample(buf):
    ar = np.frombuffer(buf, dtype=np.uint8).reshape(-1,5)
    diff = ar[1:,-1].astype(int) - ar[:-1,-1].astype(int)
    start = np.where(diff == 127)[0][0]
    return ar[start+1:start+257], True

def get_sample(l):
    S.reset_input_buffer()
    b = S.read(256*5*3)
    sample_end = time()
    try:
        s = Sample(b, sample_end)
        print(s.data.shape)
        # print(s.data[:10])
        l.append(s)
    except:
        print("bit alignment fucked")
        S.reset_input_buffer()


if __name__ == "__main__":
    samples = []
    print("SOUND RECORDING START")
    while len(samples) < 1000:
        get_sample(samples)
        if not len(samples) == 0 and len(samples) % 10 == 0:
            print(f"processed {len(samples)} samples")
        if not len(samples) == 0 and len(samples) % 200 == 0:
            fname = f"audio_samples/freqdomain_{time()}.csv"
            print("saving file...")
            with open(fname, 'w') as f:
                for s in samples:
                    f.write(s.serialize())



