import torch
from torch import tensor, Tensor, norm
from torch.nn import Module, Conv1d, Sequential, MaxPool1d, ReLU, Linear, Sigmoid, BatchNorm1d
from torch.nn.functional import mse_loss, cosine_embedding_loss, huber_loss
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from dataset import AudioPositionDataset

MAX_EPOCHS=200
MODEL_NAME="trained_models/conv1d.pth"

class DroneDirectionFinder(Module):

    def __init__(self, in_channels=4, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.to(device)
        self.encoder = Sequential(
            Conv1d(in_channels, 64, kernel_size=8, stride=2),
            ReLU(),
            MaxPool1d(4),
            Conv1d(64, 128, 4),
            ReLU(),
            BatchNorm1d(128)
        )
        self.coordinate_head = Sequential(
            Linear(12*128, 3),
            ReLU()
        )
        self.cosine_head = Sequential(
            Linear(12*128, 3),
            Sigmoid()
        )
    
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = tensor(x)
        latent = self.encoder(x)
        return self.coordinate_head(latent.reshape(-1,12*128)), self.cosine_head(latent.reshape(-1,12*128))

def run_test(model, test_dl):
    dist_errors = []
    dir_errors = []
    for fbatch, label in test_dl:
        lcoords = label[:,:3]
        lcos = label[:,3:]
        icoords, icos = model(fbatch)
        dist_errors.append(norm(icoords - lcoords))
        dir_errors.append(abs(icos - lcos).mean())
    return torch.stack(dist_errors).mean().item(), torch.stack(dir_errors).mean().item()


def train_model(model):

    opt = Adam(model.parameters(), 1e-4)

    train_ds = AudioPositionDataset(slice=slice(1700))
    test_ds = AudioPositionDataset(slice=slice(1700, None, 1))
    
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=64)
    test_dl = DataLoader(test_ds, shuffle=True, batch_size=64)

    for epoch in range(1,MAX_EPOCHS):

        print(f"Starting epoch {epoch}")

        for fbatch, label in train_dl:
            lcoords = label[:,:3]
            lcos = label[:,3:]
            coords, cosines = model(fbatch)

            # coord_loss = mse_loss(coords, lcoords)
            cos_loss = huber_loss(cosines, lcos)
            # loss = coord_loss + cos_loss

            opt.zero_grad()
            cos_loss.backward()
            # loss.backward()
            opt.step()
        
        dist_err, dir_err = run_test(model, test_dl)
        print(f"Epoch {epoch} edist {dist_err} edir {dir_err}")
    


if __name__ == "__main__":

    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = DroneDirectionFinder(device=dev)
    model.train()

    train_model(model)

    torch.save(model.state_dict(), MODEL_NAME)


