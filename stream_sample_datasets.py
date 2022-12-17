import numpy as np
import torch

from math import floor

EPS = np.finfo(float).eps

class MeanSegmentedDataset():

    """
    Time-frequency window dataset with mean-reduced position labels:
    - data format 
    
                [ ------ fft (t) ----------------- ] x 3 channels

                [ ------ fft (t + 1) ------------- ] x 3 channels
                  
                 .................................

                [ ------ fft (t + window_width) -- ] x 3 channels

    - label format [ x | y | z ]
    """

    def __init__(self, bases=["stream_samples"], suffixes=["1671276025.1548362"], device=torch.device('cpu')):

        self.device = device

        sampleses = []
        positionses = []

        for base, suffix in zip(bases, suffixes):
            sampleses.append(np.load(f"{base}/samples_{suffix}.npy"))
            positionses.append(np.load(f"{base}/positions_{suffix}.npy"))
        
        samples = np.concatenate(sampleses)
        positions = np.concatenate(positionses)

        self.data, self.labels = self.segment_dataset_by_mean(samples, positions)
        assert(len(self.data) == len(self.labels))
        self._len = len(self.data)

    @staticmethod
    def segment_dataset_by_mean(samples, positions, view_width=5):
        
        n_views = floor( samples.shape[2] / view_width )
        stop = n_views * view_width

        print(f"At view width of {view_width} dataset is at {stop / samples.shape[2] * 100}% of original size")

        data_shape = samples.shape[:2] + (n_views, view_width) + samples.shape[3:]
        data_shape_merged = (data_shape[0] * data_shape[2], data_shape[1], data_shape[3], data_shape[4])
        
        positions_shape = positions.shape[:1] + (n_views, view_width) + positions.shape[2:]
        positions_shape_merged = (positions_shape[0] * positions_shape[1], positions_shape[3])

        data = np.transpose(samples[:,:,:stop].reshape(data_shape), axes=(0,2,1,3,4)).reshape(data_shape_merged)
        labels = positions[:,:stop].reshape(positions_shape).mean(axis=2).reshape(positions_shape_merged)

        return torch.Tensor(data), torch.Tensor(labels)

    def to(self, device):
        self.device = device

    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        return (self.data[idx].to(self.device), self.labels[idx].to(self.device))

    def get_channels(self):
        return self.data.shape[1]
    
    def get_view_width(self):
        return self.data.shape[2]
    
    def get_sample_width(self):
        return self.data.shape[3]
    
    def get_label_shape(self):
        return self.labels.shape[-1]


class MeanSegmentedDatasetCosDist(MeanSegmentedDataset):

    """
    Time-frequency window dataset with mean-reduced direction and magnitude labels:
    - data format

                [ ------ fft (t) ----------------- ] x 3 channels
                
                [ ------ fft (t + 1) ------------- ] x 3 channels
                
                .................................
                
                [ ------ fft (t + window_width) -- ] x 3 channels

    - label format [ direction cosines | magnitude ]
    """

    def __getitem__(self, idx):
        data, label = super().__getitem__(idx)
        magnitude = torch.norm(label)
        label = torch.cat( [ label / magnitude + EPS, magnitude.reshape(1) ] )
        return (data, label)
    
    def get_label_shape(self):
        return self.labels.shape[-1] + 1


class MeanSegmentedDatasetCosDistHorizontal(MeanSegmentedDataset):

    """
    Time-frequency window dataset with mean-reduced direction and magnitude labels only in the horizontal plane:
    - data format

                [ ------ fft (t) ----------------- ] x 3 channels
                
                [ ------ fft (t + 1) ------------- ] x 3 channels
                
                .................................
                
                [ ------ fft (t + window_width) -- ] x 3 channels

    - label format [ direction cosines (x and y only) | magnitude (x and y only) ]
    """

    def __getitem__(self, idx):
        data, label = super().__getitem__(idx)
        label = label[:-1]
        magnitude = torch.norm(label)
        label = torch.cat( [ label / magnitude + EPS, magnitude.reshape(1) ] )
        return (data, label)

    def get_label_shape(self):
        return self.labels.shape[-1] - 1 + 1


if __name__ == "__main__":
    ds = MeanSegmentedDatasetCosDistHorizontal()
    ds.to(torch.device('cuda'))

    d, l = ds[0]
    print(len(ds))

