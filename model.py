import torch
from torch import tensor, Tensor, norm
from torch.nn import Module, Conv1d, Sequential, MaxPool1d, ReLU, Linear, Sigmoid, BatchNorm1d, Softmax
from torch.nn.functional import mse_loss, cosine_embedding_loss, huber_loss, cross_entropy
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from dataset import AudioPositionDataset

MAX_EPOCHS=2000
MODEL_NAME="trained_models/conv1d_classifier-x-dumb.pth"

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
        self.encoder = Sequential(
            Linear(516, 128),
            ReLU(),
            BatchNorm1d(128)
        )
        self.classifier_head = Sequential(
            Linear(128, 2),
            Softmax()
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
        latent = self.encoder(x.reshape(-1,516))
        return self.classifier_head(latent)
        return self.classifier_head(latent.reshape(-1,12*128))
        return self.coordinate_head(latent.reshape(-1,12*128)), self.cosine_head(latent.reshape(-1,12*128))

def run_test(model, test_dl):
    dist_errors = []
    dir_errors = []
    correct_count = 0
    total_count = 0
    for fbatch, label in test_dl:
        # lcoords = label[:,:3]
        # lcos = label[:,3:]
        # icoords, icos = model(fbatch)
        # dist_errors.append(norm(icoords - lcoords))
        # dir_errors.append(abs(icos - lcos).mean())
        probs = model(fbatch)
        total_count += len(fbatch)
        correct_count += torch.where(torch.argmax(probs, dim=-1) == torch.argmax(label,dim=-1), 1, 0).sum()
    return ( correct_count / total_count )
    return torch.stack(dist_errors).mean().item(), torch.stack(dir_errors).mean().item()


def train_model(model):

    opt = Adam(model.parameters(), 1e-4)

    train_ds = AudioPositionDataset(slice=slice(1200))
    test_ds = AudioPositionDataset(slice=slice(1200, None, 1))
    
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=32)
    test_dl = DataLoader(test_ds, shuffle=True, batch_size=32)

    for epoch in range(1,MAX_EPOCHS):

        print(f"Starting epoch {epoch}")
        
        lsum = tensor([0.])

        for fbatch, label in train_dl:
            # lcoords = label[:,:3]
            # lcos = label[:,3:]
            # coords, cosines = model(fbatch)

            # # coord_loss = mse_loss(coords, lcoords)
            # cos_loss = huber_loss(cosines, lcos)
            # # loss = coord_loss + cos_loss
            prob = model(fbatch)
            loss = cross_entropy(prob, label)

            opt.zero_grad()
            # cos_loss.backward()
            loss.backward()
            opt.step()
            lsum += loss
        
        # dist_err, dir_err = run_test(model, test_dl)
        # print(f"Epoch {epoch} edist {dist_err} edir {dir_err}")
        err = run_test(model, test_dl) * 100
        trainerr = run_test(model, train_dl) * 100
        print(f"Epoch {epoch} trainaccuracy {trainerr.item()}% validation accuracy {err.item()}%")
    


if __name__ == "__main__":

    # dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # model = DroneDirectionFinder(device=dev)
    model = DroneDirectionFinder()
    model.train()

    train_model(model)

    torch.save(model.state_dict(), MODEL_NAME)


