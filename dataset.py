import numpy as np
from torch import tensor

class AudioPositionDataset():

    def __init__(self, audio="extracted/trimmed_audio.npy", position="extracted/trimmed_position.npy"):
        self.features = np.load(audio)
        self.labels = np.load(position)
        assert(len(self.features) == len(self.labels))
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        f = tensor(self.features[idx])
        l = tensor(self.labels[idx])
        return f, l

if __name__ == "__main__":
    ds = AudioPositionDataset()
    f, l = ds[0]
    print()