import numpy as np
from torch import tensor, dot, stack, norm, concat

class AudioPositionDataset():

    def __init__(self, audio="extracted/trimmed_audio.npy", position="extracted/trimmed_position.npy", slice=slice(None)):
        features = np.load(audio).astype(np.float32)
        labels = np.load(position).astype(np.float32)
        assert(len(features) == len(labels))
        self.features = features[slice]
        self.labels = labels[slice]

    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = tensor(self.features[idx][1:-1]).reshape(-1,4).transpose(1,0)
        coords = tensor(self.labels[idx][1:])
        cosines = stack(
            [
                dot(coords, tensor([1.,0.,0.])) / norm(coords),
                dot(coords, tensor([0.,1.,0.])) / norm(coords),
                dot(coords, tensor([0.,0.,1.])) / norm(coords)
            ]
        )
        return features, concat([coords, cosines])


if __name__ == "__main__":
    ds = AudioPositionDataset()
    f, l = ds[75]
    print()